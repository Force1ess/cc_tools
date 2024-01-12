# CC_TOOl

用途：将数据从WARC中提取出来，并使用正则表达式进行初步清理，并将其根据domain分散
---

## 使用

### [build]

```
cargo build --release
```

### [run]

```
cargo run -- input_dir output_dir chunksize memory_bound
```

or

```
./target/release/cc_tool input_dir output_dir chunksize memory_bound
```

example: cargo run warc_path output_path 55 50

chunksize: 每次并行处理的文件数量，建议根据内存大小设置为 (cpu核数-4) 的倍数
chunksize 应该尽可能的大，因为一个chunk内的数据会被合并读写

memory_bound: 触发写等待内存上限，建议不能小于chunksize

---

## 系统设计

使用rayon对chunksize个文件进行多线程处理然后统一收集

使用一个单独的线程负责将收集到的数据转化为RecordBatch

并将其传给一个tokio thread进行异步处理，将数据分割成chunk，使用apache AsyncArrowWriter，将处理结果写入文件

目的是合并读写，减少IO操作，提高效率

---

## 性能

测试环境：Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz 64核

测试数据：CC-MAIN-2019-08

测试结果：

chunksize: 55

memory_bound: 50

处理100G文件，20分钟内完成，未出现明显的io bound

---

## 优化和踩坑

1. spawn出tokio的线程不能是计算密集型任务的，否则会导致被spawn出的thread得不到调度
2. AsyncArrowWriter的不应该分配缓存，因为只写一次
3. Rust调用时出错没有错误信息，出现NulError，后发现是外部接口FFI导致的
