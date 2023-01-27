#! /bin/bash

while true; do    
    BAD="$(kubectl get pods | grep NotReady | awk '{print $1}' | xargs) $(kubectl get pods | grep Error | awk '{print $1}' | xargs)"

    if [ ${#BAD} -ge 5 ]; then
        kubectl delete pod $BAD
    fi

    sleep 10
done
