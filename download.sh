while read -r line; do
    arr=($line)
    mkdir -p $(dirname ${arr[0]})
done < download_links

while read -r line; do
    arr=($line)
    curl -L -o ${arr[0]} ${arr[1]}
done < download_links
