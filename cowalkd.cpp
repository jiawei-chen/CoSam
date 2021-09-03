// cppimport
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<map>
#include<algorithm>
#include<vector>
#include<cmath>
using namespace std;
namespace py = pybind11;
#define maxn 1000005
using namespace std;
    vector<int>q1[maxn];
    vector<double>w1[maxn];
    vector<int>id1[maxn];
    vector<int>co[maxn];
    vector<int>q2[maxn];
    vector<int>id2[maxn];
    vector<double>w2[maxn];
    double w0[maxn];
    int snum[maxn];
    double eps=1e-4;
    int n,m,ld;
    double c[3];
#define maxans 20000000
struct node
{
    int x;
    int y;
}v;
vector<node>vq;

double *ran;
double *gl;
int ranmax;
int rannow;
vector<int>path;
int find(int x,int flag,int ans)
{
    int as=0;
    while(1)
    {
      // if(ans==0)
      // {
      //   cout<<x<<'h'<<flag<<endl;
      // }
        as++;
        if(as>ld)
        {
          double g=*(ran+rannow);
          rannow=(rannow+1)%ranmax;
          x=int(g*m-eps);
          path.clear();
          return x;
        }
  //      if(ans==5)
  //          cout<<x<<'h'<<flag<<endl;
        double g=*(ran+rannow);
        rannow=(rannow+1)%ranmax; 
    //    cout<<x<<'h'<<flag<<endl;
     
        if(g>c[flag])
        {
            if(flag==1)
            {
                double g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
                x=int(g*m-eps);
                path.clear();
                return x;
            }
            else
            {
                return x;
            }
        }
        if(flag==1)
        {
            flag=2;
            if(q1[x].size()==0)
            {
                double g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
                 path.push_back(2*1e8+x);
                x=int(g*m-eps);
                continue;
            }

            g=*(ran+rannow);
            rannow=(rannow+1)%ranmax;
            int p=-1;
            for(int l=0;l<w1[x].size();l++)
            {
                if(w1[x][l]>=g-eps)
                {
                    p=l;
                    break;
                }
            }
            if(p==-1)
            {
               double g=*(ran+rannow);
                rannow=(rannow+1)%ranmax;
                 path.push_back(2*1e8+x);
                x=int(g*m-eps);
                continue;
            }
            else
            {
              // if(ans==0)
              // {
              //   cout<<x<<'q'<<p<<' '<<id1[x][p]<<' '<<q1[x][p]<<endl;
              // }
                path.push_back(id1[x][p]);
                x=q1[x][p];
            }
        }
        else
        {
            flag=1;
            if(q2[x].size()==0)
            {
                throw std::runtime_error("error w2+=1"); 
            }

            g=*(ran+rannow);
            rannow=(rannow+1)%ranmax;
            int p=-1;
            for(int l=0;l<w2[x].size();l++)
            {
                if(w2[x][l]>=g-eps)
                {
                    p=l;
                    break;
                }
            }
       //     cout<<p<<endl;
            if(p==-1)
            {
              throw std::runtime_error("error w2+=1"); 
            }
            else
            {
              //  if(ans==0)
              // {
              //   cout<<x<<'a'<<p<<' '<<id2[x][p]<<' '<<q2[x][p]<<endl;
              // }
                path.push_back(1e8+id2[x][p]);
                x=q2[x][p];
            }
                
        }
       
    }
}
vector<int>anspa;
vector<int>anspid;
int ans; 
void sample(void)
{
    ans=0;
    anspa.clear();
    anspid.clear();
    rannow=0;
    vq.clear();
    for(int i=0;i<n;i++)
    {
        for(int j=1;j<w1[i].size();j++)
        {
            w1[i][j]+=w1[i][j-1];
        }
    }

    for(int i=0;i<m;i++)
    {
        for(int j=1;j<w2[i].size();j++)
        {
            w2[i][j]+=w2[i][j-1];
        }
        if(w2[i].size()>0&&w2[i][w2[i].size()-1]<1-eps)
        {
       //     cout<<i<<' '<<w2[i][w2[i].size()-1]<<endl;
             throw std::runtime_error("error w2.sum"); 
        }
    }
 //   mexPrintf("gg\n");

    for(int i=0;i<n;i++)
    {
      //   cout<<i<<endl;
        for(int ep=1;ep<=snum[i];ep++)
        {
          //  cout<<ep<<endl;
            path.clear();
         //   cout<<i<<endl;
            int y=find(i,1,ans);
        //    cout<<y<<'g'<<endl;
          //   mexPrintf("%d %d %d\n",j,i,y);
            v.x=i;
            v.y=y;
            ans++;
            vq.push_back(v);
            for(int j=0;j<path.size();j++)
            {
          //      if(ans<=10)
         //       cout<<ans<<' '<<path[j]<<endl;
                anspid.push_back(ans);
                anspa.push_back(path[j]);
            }
            if(ans>maxans)
                throw std::runtime_error("error too many output"); 
            
        }
    }
}

py::array_t<long long> negf (py::array_t<double>& input1, py::array_t<double>& input2,py::array_t<double>& input3, py::array_t<long long>& input4,py::array_t<double>& input5,py::array_t<long long>& input6,py::array_t<double>& input7) {
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    py::buffer_info buf3 = input3.request();
    py::buffer_info buf4 = input4.request();
    py::buffer_info buf5 = input5.request();
    py::buffer_info buf6 = input6.request();
    py::buffer_info buf7 = input7.request();

    int nn1=buf1.shape[0];
    int nn2=buf2.shape[0];
    int nn4=buf4.shape[0];
    ranmax=buf7.shape[0];
    if(nn1!=nn2||nn1!=nn4||buf4.shape[1]!=2||buf5.shape[0]!=5)
       throw std::runtime_error("error input");
    

  //  cout<<nn1<<'\t'<<nn2<<endl;

    auto weight1 = input1.unchecked<1>();
    auto weight2 = input2.unchecked<1>();
     auto weight0  = input3.unchecked<1>();
     auto re=input4.unchecked<2>();
    auto canshu = input5.unchecked<1>();
    auto samnum= input6.unchecked<1>();

    n=canshu(0);
    m=canshu(1);
    c[1]=canshu(2);
    c[2]=canshu(3);
    ld=canshu(4);
    if(n!=buf3.shape[0]||n!=buf6.shape[0])
        throw std::runtime_error("error n");

    ran=(double *)buf7.ptr;
    for(int i=0;i<n;i++)
    {
        snum[i]=samnum(i);
    }
    for(int i=0;i<n;i++)
    {
        q1[i].clear();
        w1[i].clear();
        co[i].clear();
        id1[i].clear();
    }
    for(int i=0;i<m;i++)
    {
        q2[i].clear();
        w2[i].clear();
        id2[i].clear();
    }
    
    for(int i=0;i<nn1;i++)
     {
        int x=re(i,0);
        int y=re(i,1);
        double z=weight1(i);
        if(x>=n||y>=m||x<0||y<0)
              throw std::runtime_error("error re"); 
        q1[x].push_back(y);
        w1[x].push_back(z);
        id1[x].push_back(i);

        double zf=weight2(i);
        q2[y].push_back(x);
        w2[y].push_back(zf);
        id2[y].push_back(i);

         // mexPrintf("%d %d\n",x,y);
     }

     
     for(int i=0;i<n;i++)
     {
        w0[i]=weight0(i);
     }
    
   
 //    cout<<nn1<<'\t'<<nn2<<endl;
   //   cout<<n<<' '<<m<<endl;
     sample();
   //  int hh=1+ans+anspa.size();
   //   auto result = py::array_t<long long>(2*hh);
   //    py::buffer_info buf8 = result.request();   
  
   //  result.resize({hh,2});
   //   auto p10=result.mutable_unchecked<2>();
   //   p10(0,0)=ans;
   //   p10(0,1)=anspa.size();
   //  for(int i=0;i<ans;i++)
   //  {
   //      v=vq[i];
   //     p10(i+1,0)=v.x;
   //     p10(i+1,1)=v.y;
   //  }
   //  for(int i=0;i<anspa.size();i++)
   //  {
   //     p10(i+ans+1,0)=anspid[i];
   //     p10(i+ans+1,1)=anspa[i];
   //  }

   // return result;
      for(int i=0;i<anspid.size();i++)
      {
        int j;
        for(j=i;j<anspid.size()-1&&anspid[j]==anspid[j+1];j++);
        int id=anspid[i]-1;
        int flag=0;
        int x=vq[id].x;
        for(int k=i;k<=j;k++)
        {

            int p=anspa[k];
            int nf=p/int(1e8);
            p=p%int(1e8);
            // if(i==0)
            // cout<<x<<' '<<flag<<' '<<p<<' '<<nf<<endl;
            if(nf%2!=flag)
               throw std::runtime_error("error nf"); 
            if(nf==0)
            {
              if(p<0||p>=nn1||re(p,0)!=x)
              {
                 cout<<x<<' '<<flag<<' '<<p<<' '<<nf<<endl;
                 throw std::runtime_error("error x1"); 
              }
              flag=1;
              x=re(p,1);
            }
            else if(nf==1)
            {
              if(p<0||p>=nn1||(x!=-1&&x!=re(p,1)))
                 throw std::runtime_error("error x2"); 
              flag=0;
              x=re(p,0);
            }
            else
            {
               if(x!=p)
                 throw std::runtime_error("error x3");
                flag=1;
                x=-1; 
            }
        }
        if(x!=-1&&x!=vq[id].y)
          throw std::runtime_error("error x4");
        i=j;

      }


     int hh=1+ans+anspa.size();
     auto result = py::array_t<long long>(2*hh);
      py::buffer_info buf10 = result.request();   
     result.resize({hh,2});
      auto p10=result.mutable_unchecked<2>();
     p10(0,0)=ans;
     p10(0,1)=anspa.size();
    for(int i=0;i<ans;i++)
    {
        v=vq[i];
       p10(i+1,0)=v.x;
       p10(i+1,1)=v.y;
    }
    for(int i=0;i<anspa.size();i++)
    {
       p10(i+ans+1,0)=anspid[i];
       p10(i+ans+1,1)=anspa[i];
    }

   return result;
    
    
    
    
}








PYBIND11_MODULE(cowalkd, m) {

    m.doc() = "Simple demo using numpy!";
    m.def("negf", &negf);
}




/*
<%
setup_pybind11(cfg)
%>
*/









