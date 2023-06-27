Scripts from Elena to facilitate setting up SSD drives and swap space to work with a Spark cluster.

Elena recommends using r5dn.4xlarge, which comes with two SSD drives that we can use for temp storage.\
**Update**: Elena mentioned that she now uses a r5dn.8xlarge machine.

You run [init_ec2_4xlarge.sh](init_ec2_4xlarge.sh), which calls the two other scripts.
You may need to modify the SSD names in that script. 
Those are "nvme2n" and "nvme1n1" for me. You can fidn the names of yours by running `lsblk`.

Here is the code used to set up the spark cluster in a notebook. 
Replace the first line of code with the path to your spark.

The commented out section on PYSPARK_SUBMIT_ARGS can be replaced by getting the appropriate jars directly from the maven repos and saving them in the `jars` subdirectory of `SPARK_HOME`.

```
## The below was copied from Elena's notebook
# Installed pyspark via conda. 
os.environ["SPARK_HOME"] = "/home/ubuntu/anaconda3/envs/2022_02_02_SRT_blueprint_exploration_and_dmr_identification_EKT/lib/python3.7/site-packages/pyspark"
# The JAR versions specified here should match spark installation used.
# Version dependencies can be found in Maven Repositories
# https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-aws/3.1.2
## os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages {aws_java},{aws_hadoop} pyspark-shell".\
##    format(aws_java="com.amazonaws:aws-java-sdk-bundle:1.11.271",
##           aws_hadoop="org.apache.hadoop:hadoop-aws:3.1.2")

# for pyspark 3.3.1, I saved the following in the `jars subdirectory`
# https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
# https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.330/aws-java-sdk-bundle-1.12.330.jar

spark_conf = SparkConf()
spark_conf.set("spark.executor.instances", "2")
spark_conf.set("spark.executor.cores", "2")
spark_conf.set("spark.executor.memory", "16g")
spark_conf.set("spark.driver.memory", "64g")
spark_conf.set("spark.driver.maxResultSize", "32g")
spark_conf.set("spark.local.dir", "/temp")
spark_conf.getAll()

sc = SparkContext(conf=spark_conf)
spark = SparkSession(sc)
```

