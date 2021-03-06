# wget

## wget的基本语法：
wget [参数列表] URL

## 1. 基础：下载整个http或者ftp站点文件
wget http://place.your.url/here

这个最简单的命令可以将http://place.your.url/here 首页下载下来。

注： 
* 使用 -x 会强制建立服务器上一模一样的目录
* 使用 -nd 参数，那么服务器上下载的所有内容都会加到本地当前目录。

### 1.1 递归下载目录里的多个文件
wget -r http://place.your.url/here

这个命令会按照递归的方法，下载服务器上所有的目录和文件，实质就是下载整个网站。

这个命令一定要小心使用，因为在下载的时候，被下载网站指向的所有地址 同样会被下载，因此，如果这个网站引用了其他网站，那么被引用的网站也会被下载下来！
基于这个原因，这个参数不常用。可以用 -l number 参数来指定下载的层次。
例如只下载两层，那么可以使用 -l 2。

制作镜像站点，那么可以使用 －m 参数，例如：

wget -m http://place.your.url/here

这时 wget 会自动判断合适的参数来制作镜像站点。此时，wget 会登录到服务器上，读入 robots.txt 并按robots.txt 的规定来执行。

## 2. 断点续传
当文件特别大或者网络特别慢的时候，往往一个文件还没有下载完，连接就已经被切断，此时就需要断点续传。

wget的断点续传是自动的，只需要使用-c参数，例如：

wget -c http://the.url.of/incomplete/file

使用断点续传要求服务器支持断点续传。
* -t 参数表示重试次数，例如需要重试100次，那么就写  -t 100，如果设成  -t 0，那么表示无穷次重试，直到连接成功。
* -T 参数表示超时等待时间，例如 -T 120，表示等待120秒连接不上就算超时。

## 3. 批量下载多个地址
如果有多个文件需要下载，那么可以生成一个文件，把每个文件的URL写一行。

例如生成文件 download.txt，然后用命令：

wget -i download.txt

这样就会把 download.txt 里面列出的每个URL都下载下来。（如果列的是文件就下载文件，如果列的是网站，那么下载首页）

## 4. 选择性的下载
可以指定让wget只下载一类文件，或者不下载什么文件。例如：

wget -m –reject=gif http://target.web.site/subdirectory

表示下载 http://target.web.site/subdirectory，但是忽略gif文件。
* –accept=LIST 可以接受的文件类型，
* –reject=LIST 拒绝接受的文件类型。

## 5. 密码和认证
wget 只能处理利用用户名/密码方式限制访问的网站，可以利用两个参数：
* –http-user=USER 设置HTTP用户
* –http-passwd=PASS 设置HTTP密码

对于需要证书做认证的网站，就只能利用其他下载工具了，例如 curl。

## 6. 利用代理服务器进行下载
如果用户的网络需要经过代理服务器，那么可以让 wget 通过代理服务器进行文件的下载。

此时，需要在当前用户的目录下创建一个 .wgetrc 文件。

文件中可以设置代理服务器：  
http-proxy = 111.111.111.111:8080 ftp-proxy = 111.111.111.111:8080  
分别表示http的代理服务器和ftp的代理服务器。

如果代理服务器需要密码则使用：  
–proxy-user=USER设置代理用户   
–proxy-passwd=PASS设置代理密码  
这两个参数。

使用参数 –proxy=on/off 使用或者关闭代理。

## 7. 后台下载
加入参数 -b， 让wget在后台运行，记录文件写在当前目录下“wget-log”文件中；

此时，最好也加上限制等待时间的 -T，和限制重试次数的 -t 参数，具体可以参照断点续传的部分。

## 附录
### A. 命令格式
wget [参数列表] [目标软件、网页的网址]
### B. 参数大全
这个地方可能还不太全，后续用到了再在本文中补充。
|参数|含义|
|---|---|
|-V,–version|显示软件版本号然后退出；
|-h,–help|显示软件帮助信息；
|-e,–execute=COMMAND|执行一个 “.wgetrc”命令
|-o,–output-file=FILE|将软件输出信息保存到文件；
|-a,–append-output=FILE|将软件输出信息追加到文件；
|-d,–debug|显示输出信息；
|-q,–quiet|不显示输出信息；
|-i,–input-file=FILE|从文件中取得URL；
|-t,–tries=NUMBER|是否下载次数（0表示无穷次）
|-O –output-document=FILE|下载文件保存为别的文件名
|-nc, –no-clobber|不要覆盖已经存在的文件
|-N,–timestamping|只下载比本地新的文件
|-T,–timeout=SECONDS|设置超时时间
|-Y,–proxy=on/off|关闭代理
|-nd,–no-directories|不建立目录
|-x,–force-directories|强制建立目录
|–http-user=USER|设置HTTP用户
|–http-passwd=PASS|设置HTTP密码
|–proxy-user=USER|设置代理用户
|–proxy-passwd=PASS|设置代理密码
|-r,–recursive|下载整个网站、目录（小心使用）
|-l,–level=NUMBER|下载层次
|-A,–accept=LIST|可以接受的文件类型
|-R,–reject=LIST|拒绝接受的文件类型
|-D,–domains=LIST|可以接受的域名
|–exclude-domains=LIST|拒绝的域名
|-L,–relative|下载关联链接
|–follow-ftp|只下载FTP链接
|-H,–span-hosts|可以下载外面的主机
|-I,–include-directories=LIST|允许的目录
|-X,–exclude-directories=LIST|拒绝的目录
