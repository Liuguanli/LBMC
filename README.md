## LBMC


### E1:

#### The verification of the correctness of our cost modelling
[./python/verify_cost.py](https://github.com/Liuguanli/LBMC/tree/main/python/verify_cost.py)

The key idea of verification is to proof that our proposed cost algorithms can get exactly the same results.

In `verify_cost.py` the following code snippet will verify the results of the cost algorithm.

```python
    assert my_gc == naive_gc, ("wrong global cost calculation my_gc:%d, naive_gc:%d", (my_gc, naive_gc))
    assert my_gc_all == naive_gc, ("wrong global cost calculation my_gc_all:%d, naive_gc:%d", (my_gc_all, naive_gc))
    assert my_lc == naive_lc, ("wrong local cost calculation my_lc:%d, naive_lc:%d", (my_lc, naive_lc))
    assert my_lc_all == naive_lc, ("wrong local cost calculation my_lc_all:%d, naive_lc:%d", (my_lc_all, naive_lc))
```

#### How to calculate drop patterns and rise patterns:
Please refer to [```calculate_drop_pattern```](
https://github.com/Liuguanli/LBMC/tree/main/python/utils.py#L50)
and [```calculate_rise_pattern```](https://github.com/Liuguanli/LBMC/tree/main/python/utils.py#L64).



#### Cost for *n* queries and *m* BMCs:

Global Cost: please refer to [```global_cost```](https://github.com/Liuguanli/LBMC/tree/main/python/global_cost.py#L115)

Please refer to formula (5).


Local Cost: Please refer to [```local_cost```](https://github.com/Liuguanli/LBMC/tree/main/python/local_cost.py#L137)

Please refer to Algorithm 1.

### E2:

#### Datasets

All used datasets are listed: [here](https://drive.google.com/drive/folders/1RK1SuFumCTpHrlyL22zEWV6qgdYD07Vs)

#### Integrate cost estimations to BMTree

Please refer to [```Learned-BMTree```](https://github.com/Liuguanli/LBMC/tree/main/Learned-BMTree/utils/metric_compute.py#L184)

<!-- This part is for Section 6.3. -->


#### Comparison via PostgreSQL

For BMTree, please refer to [```Learned-BMTree pg_test.py```](https://github.com/Liuguanli/LBMC/tree/main/Learned-BMTree/pg_test.py)

For others, please refer to [```LearnSFC pg_test.py```](https://github.com/Liuguanli/LBMC/tree/main/python/pg_test.py)
<!-- 
This part is used in Section 6.4. If you want to run the code, please add the [datasets](https://drive.google.com/drive/folders/15fTAbMIuJSNF1o3t36NODuaahtt3O7IV) to `./Learned-BMTree/data/` -->



### E3: 

#### Datasets

TPC:

https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp

```bash
./dbgen -s 1 -T o
```

NYC:

https://data.cityofnewyork.us/Transportation/2017-Yellow-Taxi-Trip-Data/biws-g3hs/about_data


#### Run queries on Hudi

For NYC dataset, please refer to [```nyc.scala```](https://github.com/Liuguanli/LBMC/tree/main/hudi/scala/nyc.scala)

For TPC-H dataset, please refer to [```tpc.scala```](https://github.com/Liuguanli/LBMC/tree/main/hudi/scala/tpc.scala)

