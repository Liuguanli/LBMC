#!/bin/bash

export SPARK_VERSION=3.4

export HUDI_LAYOUT="linear"
export DATA_FILE="tpc_1"
export NUM_QUERIES=100

# LBMC ABABABABABABABAB


hudi_layouts=("linear" "z-order" "hilbert" "bmc")
hudi_layouts=("bmc")
# hudi_layouts=("z-order") 

file_names=("tpc_1" "tpc_2" "tpc_4" "tpc_8" "tpc_16")
file_names=("tpc_16")

bytes=("2" "3" "4" "6" "8")

bmc_patterns=("ABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB")
bmc_patterns=("BABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABA" "ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB")

for name in "${file_names[@]}"
do

    export DATA_FILE="$name"
    export HUDI_LAYOUT="$layout"
    export HUDI_BMC_PATTERN="AB"
    export BYTES_NUM="8"

    for layout in "${hudi_layouts[@]}"
    do
        export HUDI_LAYOUT="$layout"

        if [ "$layout" = "bmc" ]; then
            for bmc_pattern in "${bmc_patterns[@]}"
            do
                export HUDI_BMC_PATTERN="$bmc_pattern"
                {
                    echo ":load ./hudi/scala/tpc.scala"
                } | 
                spark-shell --driver-memory 48g \
                --executor-memory 48g \
                --jars /home/research/Dropbox/projects/hudi/packaging/hudi-spark-bundle/target/hudi-spark$SPARK_VERSION-bundle_2.12-0.14.0.jar \
                --conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
                --conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog' \
                --conf 'spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension' \
                --conf 'spark.kryo.registrator=org.apache.spark.HoodieSparkKryoRegistrar' \
                --packages ch.cern.sparkmeasure:spark-measure_2.12:0.23 
            done
        elif [ "$layout" = "hilbert" ] || [ "$layout" = "z-order" ]; then
            for byte in "${bytes[@]}"
            do
                export BYTES_NUM="$byte"
                {
                    echo ":load ./hudi/scala/tpc.scala"
                } | 
                spark-shell --driver-memory 48g \
                --executor-memory 48g \
                --jars /home/research/Dropbox/projects/hudi/packaging/hudi-spark-bundle/target/hudi-spark$SPARK_VERSION-bundle_2.12-0.14.0.jar \
                --conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
                --conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog' \
                --conf 'spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension' \
                --conf 'spark.kryo.registrator=org.apache.spark.HoodieSparkKryoRegistrar' \
                --packages ch.cern.sparkmeasure:spark-measure_2.12:0.23 
            done
        else
            {
                echo ":load ./hudi/scala/tpc.scala"
            } | 
            spark-shell --driver-memory 48g \
            --executor-memory 48g \
            --jars /home/research/Dropbox/projects/hudi/packaging/hudi-spark-bundle/target/hudi-spark$SPARK_VERSION-bundle_2.12-0.14.0.jar \
            --conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
            --conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog' \
            --conf 'spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension' \
            --conf 'spark.kryo.registrator=org.apache.spark.HoodieSparkKryoRegistrar' \
            --packages ch.cern.sparkmeasure:spark-measure_2.12:0.23 
        fi
    done
done

