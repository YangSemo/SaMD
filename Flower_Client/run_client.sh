# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
# sh run.sh => 실행
#!/bin/bash

# client 수 설정
for i in `seq 0 9`; do
    echo "Starting client $i"
    python /Users/yangsemo/pycharm/news_flower/Flower_Client/client.py --n=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait