# Linux命令

## 目录
1. 统计当前目录下文件的个数（不包括目录）  
$ ls -l | grep "^-" | wc -l
2. 统计当前目录下文件的个数（包括子目录）  
$ ls -lR| grep "^-" | wc -l
3. 查看某目录下文件夹(目录)的个数（包括子目录）  
$ ls -lR | grep "^d" | wc -l

## Ubuntu如何查看版本信息
1. 使用命令： cat /proc/version 查看
2. 使用命令： uname -a 查看
3. 使用命令： lsb_release -a 查看

## 上传服务器
scp /home/work/source.txt work@192.168.0.10:/home/work/   
#把本地的source.txt文件拷贝到192.168.0.10机器上的/home/work目录下

scp work@192.168.0.10:/home/work/source.txt /home/work/   
#把192.168.0.10机器上的source.txt文件拷贝到本地的/home/work目录下

scp work@192.168.0.10:/home/work/source.txt work@192.168.0.11:/home/work/   
#把192.168.0.10机器上的source.txt文件拷贝到192.168.0.11机器的/home/work目录下

scp -r /home/work/sourcedir work@192.168.0.10:/home/work/   
#拷贝文件夹，加-r参数

## 解压缩
### .tar
解包：tar xvf FileName.tar   
tar xvf FileName.tar -C DirName（解压到指定文件夹DirName）  
打包：tar cvf FileNametar DirName
***

### .gz
解压1：gunzip FileName.gz  
解压2：gzip -d FileName.gz  
压缩：gzip FileName
### .tar.gz 和 .tgz
解压：tar zxvf FileName.tar.gz  
压缩：tar zcvf FileName.tar.gz FileName
***

### .bz2
解压1：bzip2 -d FileName.bz2  
解压2：bunzip2 FileName.bz2  
压缩： bzip2 -z FileName
### .tar.bz2
解压：tar jxvf FileName.tar.bz2  
压缩：tar jcvf FileName.tar.bz2 FileName
***

### .bz
解压1：bzip2 -d FileName.bz  
解压2：bunzip2 FileName.bz  
压缩：未知
### .tar.bz
解压：tar jxvf FileName.tar.bz  
压缩：tar jcvf FileName.tar.bz FileName
***

### .z
解压：uncompress FileName.Z  
压缩：compress FileName.tar.z  
解压：tar zxvf FileName
### .tar.z
解压：tar zxvf FileName.tar.z  
压缩：tar zcvf FileName.tar.z DirName
***

### .zip
解压：unzip FileName.zip  
压缩：zip FileName.zip DirName
***

### .rar
解压：rar x FileName.rar  
压缩：rar a FileName.rar DirName  
rar请到：http://www.rarsoft.com/download.htm 下载  
解压后请将rar_static拷贝到/usr/bin目录（其他由$PATH环境变量指定的目录也可以）[root@www2 tmp]# cp rar_static /usr/bin/rar
***

### .lha
解压：lha -e FileName.lha  
压缩：lha -a FileName.lha FileName  
lha请到：http://www.infor.kanazawa-it.ac.jp/~ishii/lhaunix/下载！  
解压后请将lha拷贝到/usr/bin目录（其他由$PATH环境变量指定的目录也可以）[root@www2 tmp]# cp lha /usr/bin/
***

### .rpm　　
解包：rpm2cpio FileName.rpm | cpio -div
***

### .deb　　
解包：ar p FileName.deb data.Tar.gz | Tar zxf -