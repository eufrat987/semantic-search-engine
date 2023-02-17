#!/bin/bash

curl --request POST \
     --url http://127.0.0.1:8000/query \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '{
     "query": "$(1)"
     }'
