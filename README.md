# CC_Tools

```bash
✗ python3 main.py -h        
usage: main.py [-h] {getpath,update,rank2domain,process,download} ...

用于CommonCrawl数据的下载与处理

positional arguments:
  {getpath,update,merge,rank2domain,process,download}
    getpath             下载的两个时间戳内的path, eg:2013/05
    rank2domain         转换ranks为domain,需要安装go
    process             处理warc.gz
    download            下载指定path中的文件

options:
  -h, --help            show this help message and exit
```

1. 获取path文件，WARC/WET可选

```bash
✗ python3 main.py getpath -h                                             
usage: main.py getpath [-h] [--start_timestamp START_TIMESTAMP] [--end_timestamp END_TIMESTAMP] [--type TYPE]

options:
  -h, --help            show this help message and exit
  --start_timestamp START_TIMESTAMP, -s START_TIMESTAMP
  --end_timestamp END_TIMESTAMP, -e END_TIMESTAMP
  --type TYPE, -t TYPE  类型: cc-news, cc-main
```
2. 从CommonCrawl的ranks文件提取domain数据~~，暂时没啥用~~
```bash
✗ python3 main.py rank2domain -h
usage: main.py rank2domain [-h] rank_path

positional arguments:
  rank_path   CommonCrawl rank地址

options:
  -h, --help  show this help message and exit
```

3. 处理WARC或WET文件

```bash
✗ python3 main.py process -h                                            
usage: main.py process [-h] [--input INPUT] [--output OUTPUT] [--type TYPE]

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        待处理文件的文件夹地址
  --output OUTPUT, -o OUTPUT
                        输出文件的文件夹地址
  --type TYPE, -t TYPE  warc or wet
```

4. 下载指定文件中的WARC/WET

```bash
✗ python3 main.py download -h
usage: main.py download [-h] [-i INPUT] [-o OUTPUT] [--num NUM]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        目标path文件
  -o OUTPUT, --output OUTPUT
                        下载位置
  --num NUM, -n NUM     同时下载的线程数
```