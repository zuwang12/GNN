#!/bin/bash

# 변수 설정
num_nodes=50
constraint_type="cluster"
beam_size=2
now=$(date +"%F_%T")

# 로그 파일 이름 설정 (num_nodes를 이용하여 동적으로 생성)
log_dir="./logs/gnn/${constraint_type}"
log_file="${log_dir}/tsp${num_nodes}_gnn_beamsize${beam_size}_${now}.log"

# 로그 파일 저장할 디렉토리가 존재하지 않으면 생성
mkdir -p "$log_dir"

# Python 스크립트 nohup으로 실행하여 백그라운드에서 동작하게 함
nohup python main_test.py --num_nodes "$num_nodes" --constraint_type "$constraint_type" --time "$now" --beam_size "$beam_size"> "$log_file" 2>&1 &
