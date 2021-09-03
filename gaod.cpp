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

using namespace std;

py::array_t<double> gao (py::array_t<double>& input1, py::array_t<double>& input2,py::array_t<double>& input3,py::array_t<double>& input4,py::array_t<long long>& input5) {
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    py::buffer_info buf3 = input3.request();
    py::buffer_info buf4 = input4.request();
    py::buffer_info buf5 = input5.request();
    

    int nn1=buf1.shape[0];
    int nn2=buf2.shape[0];
    int n=buf3.shape[0];
    int nn4=buf4.shape[0];

    if(nn1!=nn2)
       throw std::runtime_error("error input");
    

  //  cout<<nn1<<'\t'<<nn2<<endl;

    auto w1 = input1.unchecked<1>();
    auto w2 = input2.unchecked<1>();
     auto w0  = input3.unchecked<1>();
     auto reward=input4.unchecked<1>();
    auto path = input5.unchecked<2>();
    int npath=buf5.shape[0];
    int hh=2*nn1+n;
    vector<double>re(hh);
     auto result = py::array_t<double>(hh);
      py::buffer_info buf10 = result.request();   
      auto p10=result.mutable_unchecked<1>();
    for(int i=0;i<npath;i++)
    {
      int ans=path(i,0)-1;
      int flag=path(i,1);
      if(ans<0||ans>=nn4)
        throw std::runtime_error("ans");
      if(flag<1e8)
      {
        if(flag<0||flag>=nn1)
        {
          throw std::runtime_error("error w1");
        }
        re[flag]+=reward(ans)/w1(flag);
      }
      else if(flag<2e8)
      {
        if(flag>=1e8+nn1)
           throw std::runtime_error("error w2");
        re[nn1+flag-1e8]+=reward(ans)/w2(flag-1e8);
      }
      else
      {
        if(flag>=2e8+n)
          throw std::runtime_error("error w0");
         re[2*nn1+flag-2e8]+=reward(ans)/w0(flag-2e8);
      }
    }
    for(int i=0;i<hh;i++)
    {
      p10(i)=re[i];
    }

   return result;
    
    
    
    
}



PYBIND11_MODULE(gaod, m) {

    m.doc() = "Simple demo using numpy!";
    m.def("gao", &gao);
}

/*
<%
setup_pybind11(cfg)
%>
*/
