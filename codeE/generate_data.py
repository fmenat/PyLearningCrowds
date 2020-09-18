import numpy as np
import pickle

def do_gaussianEX(n=250,std=0.3):
    rng = np.random.RandomState(0)
    u = 1
    n1 = np.random.poisson(n)
    n2 = np.random.poisson(n)
    n3 = np.random.poisson(n)
    first_class = rng.normal(loc=[0,u],scale=std,size=(n1,2))
    second_class = rng.normal(loc=[0,-u],scale=std,size=(n2,2))
    third_class = rng.normal(loc=[u,0],scale=std,size=(n3,2))
    X_data = np.vstack((first_class,second_class,third_class))
    first_class = rng.normal(loc=[0,u],scale=std,size=(n1,2))
    second_class = rng.normal(loc=[0,-u],scale=std,size=(n1,2))
    third_class = rng.normal(loc=[u,0],scale=std,size=(n1,2))
    X_data = np.vstack((first_class,second_class,third_class))
    Z_data = np.hstack((np.zeros(n1),np.ones(n1),np.ones(n1)*2))  
    indexs = np.arange(Z_data.shape[0])
    np.random.shuffle(indexs)
    return X_data[indexs],Z_data[indexs]


class SyntheticData(object):
    def __init__(self,state=None):
        self.probas = False #not yet
        self.Random_num = np.random.RandomState(None)
        if type(state) ==str:
            with open(state, 'rb') as handle:
                aux = pickle.load(handle)  #read from file
                self.Random_num.set_state(aux)
        elif type(state) == tuple or type(state) == int:
            self.Random_num.set_state(state)
            
        self.init_state = self.Random_num.get_state() #to replicate

    def set_probas(self, file_matrix, file_groups, asfile = False):
        if asfile:
            load_matrix = np.loadtxt(file_matrix,delimiter=',')
            rows,self.Kl = load_matrix.shape
            self.conf_matrix = []
            for j in np.arange(self.Kl,load_matrix.shape[0]+1,self.Kl):
                self.conf_matrix.append(load_matrix[j-self.Kl:j])
            self.conf_matrix = np.asarray(self.conf_matrix)

            self.prob_groups = np.loadtxt(file_groups,delimiter=',')
        else:
            self.conf_matrix = np.asarray(file_matrix) 
            self.prob_groups = np.asarray(file_groups)
            self.Kl = self.conf_matrix.shape[1]
        self.probas = True
  
    def synthetic_annotate_data(self, Z, Tmax, T_data, deterministic=False, hard=True):
        print("New Synthetic data is being generated...",flush=True,end='')
        if not self.probas:
            self.set_probas()

        N = Z.shape[0]
        #sample group for every annotator:
        synthetic_annotators_group = []
        for t in range(Tmax):
            if hard:
                S_t = 1
            else: #soft
                S_t = max([self.Random_num.poisson(self.prob_groups.shape[0]+1),1]) 

            grupo = self.Random_num.multinomial(S_t,self.prob_groups)
            
            if hard:
                grupo = [np.argmax(grupo)]
            else:
                grupo = grupo/np.sum(grupo) #soft
            synthetic_annotators_group.append(grupo)
        synthetic_annotators_group = np.asarray(synthetic_annotators_group)
        
        synthetic_annotators = -1*np.ones((N,Tmax),dtype='int32')
        
        yo_count = np.zeros((N,self.Kl))
        prob = T_data/float(Tmax) #probability that she annotates
        for i in range(N):
            #get ground truth of data 
            if Z[i].shape != ():
                z = int(np.argmax(Z[i]))
            else:
                z = int(Z[i])

            if deterministic:
                Ti = self.Random_num.choice(np.arange(Tmax), size=T_data, replace=False) #multinomial of index
                for t in Ti: #index of annotators
                    #get group of annotators
                    g = synthetic_annotators_group[t] #in discrete value, g {0,1,...,M}
                    if hard:
                        sample_prob = self.conf_matrix[g[0],z,:]
                    else: #soft
                        sample_prob = np.tensordot(g[:], self.conf_matrix[:,z,:], axes=[[0],[0]]) #mixture
                    #sample trough confusion matrix 
                    yo = np.argmax( self.Random_num.multinomial(1, sample_prob) )
                    synthetic_annotators[i,t] = yo
                    yo_count[i,yo] +=1
            else:
                for t in range(Tmax):
                    if self.Random_num.rand() <= prob: #if she label the data i
                        #get group of annotators
                        g = synthetic_annotators_group[t] #in discrete value, g {0,1,...,M}
                        if hard:
                            sample_prob = self.conf_matrix[g[0],z,:]
                        else: #soft
                            sample_prob = np.tensordot(g[:], self.conf_matrix[:,z,:], axes=[[0],[0]]) #mixture
                        #sample trough confusion matrix 
                        yo = np.argmax( self.Random_num.multinomial(1, sample_prob) )
                        synthetic_annotators[i,t] = yo
                        yo_count[i,yo] +=1
                        
            if np.sum( synthetic_annotators[i,:] != -1)  == 0: #avoid data not labeled
                t_rand = self.Random_num.randint(0,Tmax)
                g = synthetic_annotators_group[t_rand] #in discrete value, g {0,1,...,M}
                if hard:
                    sample_prob = self.conf_matrix[g[0],z,:]
                else: #soft
                    sample_prob = np.tensordot(g[:], self.conf_matrix[:,z,:], axes=[[0],[0]]) #mixture

                synthetic_annotators[i,t_rand] = np.argmax( self.Random_num.multinomial(1, sample_prob) )
                yo_count[i,synthetic_annotators[i,t_rand]] +=1
        self.yo_label = yo_count.argmax(axis=1) #get yo_hard
        #clean the annotators that do not label
        mask_label = np.where(np.sum(synthetic_annotators,axis=0) != synthetic_annotators.shape[0]*-1)[0]
        synthetic_annotators = synthetic_annotators[:,mask_label]
        
        print("Done! ")
        return synthetic_annotators,synthetic_annotators_group[mask_label,:]

    def save_annotations(self, annotations, file_name='annotations',npy=True):
        if npy:
            np.save(file_name+'.npy',annotations.astype('int32')) 
        else:
            np.savetxt(file_name+'.csv',annotations,delimiter=',',fmt='%d') 