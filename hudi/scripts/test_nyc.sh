#!/bin/bash

export SPARK_VERSION=3.4
export NUM_QUERIES=1000

hudi_layouts=("z-order" "hilbert" "linear" "bmc")
# hudi_layouts=("z-order" "hilbert") 
# hudi_layouts=("bmc", "z-order") 
# hudi_layouts=("bmc") 

# bmc_patterns=("ABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB" "ABABABABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB")
# bmc_patterns=("BABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABA" "ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB")
bmc_patterns=("ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAAAABBBB")
bytes=("8")

for layout in "${hudi_layouts[@]}"
do
    export HUDI_LAYOUT="$layout"
    export HUDI_BMC_PATTERN="AB"
    export BYTES_NUM="8"

    # Check if the current layout is 'bmc'
    if [ "$layout" = "bmc" ]; then
        # Iterate through each BMC pattern when layout is 'bmc'
        for bmc_pattern in "${bmc_patterns[@]}"
        do
            export HUDI_BMC_PATTERN="$bmc_pattern"
            {
                echo ":load ./hudi/scala/nyc.scala"
            } | 
            spark-shell --driver-memory 48g \
            --executor-memory 48g \
            --jars /home/research/Dropbox/projects/hudi/packaging/hudi-spark-bundle/target/hudi-spark$SPARK_VERSION-bundle_2.12-0.14.0.jar \
            --conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
            --conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog' \
            --conf 'spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension' \
            --conf 'spark.kryo.registrator=org.apache.spark.HoodieSparkKryoRegistrar' \
            --packages ch.cern.sparkmeasure:spark-measure_2.12:0.23 
            # Logic here for processing with each bmc_pattern
        done
    elif [ "$layout" = "hilbert" ] || [ "$layout" = "z-order" ]; then
        for byte in "${bytes[@]}"
        do
            export BYTES_NUM="$byte"
            {
                echo ":load ./hudi/scala/nyc.scala"
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
        # Execute commands for other layouts
        {
            echo ":load ./hudi/scala/nyc.scala"
        } | 
        spark-shell --driver-memory 48g \
        --executor-memory 48g \
        --jars /home/research/Dropbox/projects/hudi/packaging/hudi-spark-bundle/target/hudi-spark$SPARK_VERSION-bundle_2.12-0.14.0.jar \
        --conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
        --conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog' \
        --conf 'spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension' \
        --conf 'spark.kryo.registrator=org.apache.spark.HoodieSparkKryoRegistrar' \
        --packages ch.cern.sparkmeasure:spark-measure_2.12:0.23 
        # Default pattern or logic for non-BMC layouts
    fi
    # export HUDI_LAYOUT="$layout"

    # export HUDI_BMC_PATTERN="ABABABABAAAABBBB"
    
    # {
    #     echo ":load ./hudi/scala/nyc.scala"
    # } | 
    # spark-shell --driver-memory 48g \
    # --executor-memory 48g \
    # --jars /home/research/Dropbox/projects/hudi/packaging/hudi-spark-bundle/target/hudi-spark$SPARK_VERSION-bundle_2.12-0.14.0.jar \
    # --conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
    # --conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog' \
    # --conf 'spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension' \
    # --conf 'spark.kryo.registrator=org.apache.spark.HoodieSparkKryoRegistrar' \
    # --packages ch.cern.sparkmeasure:spark-measure_2.12:0.23 
    # # --class ch.cern.testSparkMeasure.testSparkMeasure
done
