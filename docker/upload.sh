str=$(find data -type f  | sed 's/^/--form files=@/g')

cstr="curl --request POST \
     --url http://127.0.0.1:8000/file-upload \
     --header 'accept: application/json' \
     --header 'content-type: multipart/form-data' \
     `echo $str` \
     --form meta=null"

`$cstr`



