# CoSam

This is our source codes for the paper : <br>
CoSam: An Efficient Collaborative Adaptive Sampler for Recommendation. J Chen, C Jiang, C Wang, S Zhou, Y Feng, C Chen, M Ester, X He. ACM Transactions on Information Systems (TOIS) 39 (3), 1-24

# Example to Run CoSam
We implement CoSam in Python 3.6. The required packages are as follows:

- pytorch=1.5.1 <br>
- numpy=1.19.2 <br>
- pandas=0.20.3 <br>
- cppimport == 18.11.8 <br>
- pybind11 == 2.5.0 <br>

We can run the code for the example Ciao data: <br>
```shell
python Cosam.py --trainingdata 'trainingdata_ciao.txt' --testdata 'testdata_ciao.txt'
```
Where the inputs of the Cosam function are the paths of the trainning data and the test data. Also, you can directly use argumentparser to set the hyperparameters (e.g., --baselr XX). More details can refer to our desrciption of hyperparameters (line 79-93). <br>

Each line of trainingdata_ciao.txt is: UserID \t ItemID \t 1 <br>
Each line of testdata_ciao.txt is :UserID \t ItemID \t 1 <br>
Noting: when the number of users or items are above a certain theroshold (1000000), you need to enlarge the setting of 'maxnm' in the source code Cowalkd.cpp (line 11). Also, you may need to preprocess the dataset such that each user or item has at least one interaction. <br>

If you have any question, please feel free to contact us (E-mail: sleepyhunt@zju.edu.cn).


