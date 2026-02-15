find logs -name "*.csv" -exec bash -c 'echo "{}"; wc -l {}; echo "";' \;
