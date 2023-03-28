#!/bin/bash

status=0
current_time=0

curl http://127.0.0.1:8080/calculate_weights/0/
while [ status = 0 ]
  do
    status = curl  http://127.0.0.1:8080/status/0/
    echo $status
  done
curl http://127.0.0.1:8080/get_weights/0/