cd=1
import torch 
import pandas as pd
import numpy as np
import cppimport.import_hook
import ex
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from torch.utils.data import DataLoader, Dataset
class WMF(torch.nn.Module):
    def __init__(self, config):
        super(WMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user.weight.data*=0.1
        self.embedding_item.weight.data*=0.1
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
    def allpre(self):
        rating=self.logistic(self.embedding_user.weight.mm(self.embedding_item.weight.t()))
        return rating
    def itempre(self,samitem):
        rating=self.logistic(self.embedding_user.weight.mm(self.embedding_item(samitem).t()))
        return rating
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = torch.sum(element_product,dim=1)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class Gen(torch.nn.Module):
    def __init__(self, config,tposuser,tpositem):
        super(Gen, self).__init__()
        self.n = config['num_users']
        self.m=config['num_items']
        self.r1=torch.tensor(tposuser)
        self.r2=torch.tensor(tpositem)
        self.ct=torch.stack((self.r1,self.r2),dim=0)
        self.cf=torch.stack((self.r2,self.r1),dim=0)
        self.w1=torch.nn.Parameter(torch.Tensor(np.random.random(len(self.r1))-0.5))
        self.w2=torch.nn.Parameter(torch.Tensor(np.random.random(len(self.r1))-0.5))
        self.w0=torch.nn.Parameter(torch.Tensor(np.random.random(self.n)-0.5))
        
      
    def forward(self):
        expw1=torch.exp(self.w1)
        expw2=torch.exp(self.w2)
        expw0=torch.exp(self.w0)
        sumw1=expw0.clone()
        sumw1.scatter_add_(0,self.r1,expw1)+eps
        lfw1=expw1.div(sumw1[self.r1])
        lfw0=expw0.div(sumw1)
        if(cd==0):
            sumw2=torch.zeros((self.m),device='cpu')
        else:
            sumw2=torch.zeros((self.m),device='cuda')
        sumw2.scatter_add_(0,self.r2,expw2)+eps
        lfw2=expw2.div(sumw2[self.r2])
        return lfw0, lfw1, lfw2


import argparse
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainingdata', type=str, default='trainingdata_ciao.txt', help='The path of training data.')
    parser.add_argument('--testdata', type=str, default='testdata_ciao.txt', help='The path of test data')
    parser.add_argument('--baselr', type=float, default=0.1, help='learning rate of base RS model')
    parser.add_argument('--basedecay', type=float, default=1, help='decay of base RS model')
    parser.add_argument('--samlr', type=float, default=0.01, help='learning rate of sampling model')
    parser.add_argument('--samdecay', type=float, default=1, help='decay of sampling model')
    parser.add_argument('--topK', type=int, default=5, help='Top-k recommendation')
    parser.add_argument('--epochs', type=int, default=500, help='number of iterations')
    parser.add_argument('--emb', type=int, default=10, help='length of embeddings')
    parser.add_argument('--c1', type=float, default=1, help='c1')
    parser.add_argument('--c2', type=float, default=0.8, help='c2')
    parser.add_argument('--ld', type=int, default=4, help='max length of propagation')
    return parser.parse_args()


if __name__ == "__main__": 
    args = parse_args()
    print('Running CoSam:')



# data process

# 设置随机数种子
    #setup_seed(0)


    from scipy.sparse import csr_matrix
    config = {   'num_users': 0,
                'num_items': 0,
                'latent_dim': 10,
                }
    negconst=4
    np.random.seed(0)
    ch=args.topK
    eps=1e-8
    ld=args.ld
    data=pd.read_table(args.trainingdata,header=None)
    test=pd.read_table(args.testdata,header=None)
    n=max(data[:][0].append(test[:][0])) 
    # number of users
    m=max(data[:][1].append(test[:][1]))
    # number of items

    data=data-1;
    test=test-1;
    posuser=np.array(data[:][0])
    positem=np.array(data[:][1])
    r1=posuser
    r2=positem
    posrating=np.ones((len(posuser)))
    config['num_users']=n;
    config['num_items']=m;

    testuser=np.array(test[:][0])
    testitem=np.array(test[:][1])
    hh=np.ones(len(posuser))
    hr=csr_matrix((hh, (posuser, positem)),shape=(n, m))
    suhr=np.array(hr.sum(axis=1)).reshape(-1)

    import cowalkd
    def sampleco(w0,w1,w2,chuan,can,samnum,ran):
        zhong=cowalkd.negf(w1,w2,w0,chuan,can,samnum,ran)
        numans=zhong[0,0]
        samu=zhong[1:numans+1,0]
        sami=zhong[1:numans+1,1]
        path=zhong[numans+1:,:]
        rating=np.asarray(hr[samu,sami]).reshape(-1).astype(np.float32)
        return samu,sami,rating,path





  
    #setup_seed(0)
    import time
    import gaod
    beta1=0.9;
    beta2=0.999;
    eps=1e-8;
    constneg=4
    model = WMF(config)

    opt=torch.optim.Adam(model.parameters(),lr=args.baselr,weight_decay=args.basedecay)
    chuan=np.hstack((posuser.reshape(-1,1),positem.reshape(-1,1)))
    tposuser=torch.tensor(posuser)
    tpositem=torch.tensor(positem)
    samnum=suhr*constneg
    if(cd==1):
        model.cuda()
        tposuser=tposuser.cuda()
        tpositem=tpositem.cuda()

    nn=len(posuser)
    c1=args.c1
    c2=args.c2
    pt=1
    flag=0
    for i in range(ld):
        if(flag==0):
            pt=pt*c1
            flag=1
        else:
            pt=pt*c2
            flag=0
    gen=Gen(config,tposuser,tpositem)
    if(cd==1):
        gen.cuda()
    canshu=np.array([n,m,c1,c2,ld])
    p0=(1-c1)/(1-c1*c2)/m+pt/m
    tw0,tw1,tw2=gen()
    w1=np.array(tw1.detach().cpu())
    xuan=csr_matrix((w1*(1-c1), (posuser, positem)),shape=(n, m))

    bce=torch.nn.BCELoss()
    optgen=torch.optim.Adam(gen.parameters(),lr=args.samlr,weight_decay=args.samdecay)

    

    for ep in range(args.epochs+1):
        print('Iterations:',ep,'/',args.epochs,end='\r',flush=True)
        tw0,tw1,tw2=gen()
        w0=np.array(tw0.detach().cpu())
        w1=np.array(tw1.detach().cpu())
        w2=np.array(tw2.detach().cpu())
        ran=np.random.random((len(posuser)*100))
        user, item, rating, path= sampleco(w0,w1,w2,chuan,canshu,samnum,ran)
        spid=np.where(rating>0)
        snid=np.where(rating==0)
        traineguser=user[snid]
        trainegitem=item[snid]
        trainegrating=rating[snid]
        trainuser=torch.LongTensor(np.append(posuser,traineguser))
        trainitem=torch.LongTensor(np.append(positem,trainegitem))
        trainrating=torch.FloatTensor(np.append(posrating,trainegrating))
        if(cd==1):
            trainuser=trainuser.cuda()
            trainitem=trainitem.cuda()
            trainrating=trainrating.cuda()
        opt.zero_grad()
        pre=model(trainuser,trainitem)
        loss=bce(pre,trainrating)*len(pre)
        loss.backward()
        opt.step()

        npre=np.array(pre.detach().cpu())
        rewardneg=np.log(1-npre[nn:])
        samposuser=user[spid]
        sampositem=item[spid]

        #up xuan
        nxuan=csr_matrix((np.ones(len(samposuser)), (samposuser, sampositem)),shape=(n, m))
        xuan=xuan*beta1+(1-beta1)*nxuan
        rewardpos=1/(samnum[samposuser]*p0+xuan[samposuser,sampositem])
        reward=np.zeros(len(user))
        reward[spid]=rewardpos
        reward[snid]=rewardneg

        dw=gaod.gao(w1,w2,w0,reward,path)
        tdw1=torch.FloatTensor(dw[:nn])
        tdw2=torch.FloatTensor(dw[nn:2*nn])
        tdw0=torch.FloatTensor(dw[2*nn:])
        if(cd==1):
            tdw1=tdw1.cuda()
            tdw2=tdw2.cuda()
            tdw0=tdw0.cuda()
        lossgen=-torch.sum(tdw1*tw1)+torch.sum(tdw2*tw2)+torch.sum(tdw0*tw0)
        # print(lossgen)
        optgen.zero_grad()
        lossgen.backward()
        optgen.step()
    #  print(tw1[:10])

        #prediction
        if(ep%100==0):
            prerating=model.allpre().detach().cpu().numpy()
            cu=np.ones((n,m))*p0;
            nowfa=np.identity(n);
            sw1=csr_matrix((w1,(r1,r2)),shape=[n,m]).toarray();
            sw2=csr_matrix((w2,(r2,r1)),shape=[m,n]).toarray();
            fa0=w0.reshape(-1,1)/m;
            for i in range(ld//2):
                zhong=c1*((nowfa).dot(sw1)+nowfa.dot(fa0));
                cu=cu+zhong*(1-c2);
                nowfa=zhong.dot(sw2*c2);
            prefinal=cu*prerating
            prefinal[posuser.reshape(-1),positem.reshape(-1)]=-(1<<50)
            id=np.argsort(prefinal,axis=1,kind='quicksort',order=None)
            id=id[:,::-1]
            id1=id[:,:ch]
        # print(id1) 
            ans=ex.gaotest(testuser,testitem,id1,id)
            print('Iterations:',ep,'Performance:','Precions@K=',ans[0],'Recall@K=',ans[1],'NDCG=',ans[2])