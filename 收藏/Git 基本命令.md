# Git 基本命令
|命令|说明|
|---|---|
|git init|初始化仓库|
|git clone|拷贝一份远程仓库，也就是下载一个项目|
|git add|添加文件到仓库|
|git status|查看仓库当前的状态，显示有变更的文件|
|git diff|比较文件的不同，即暂存区和工作区的差异|
|git commit|提交暂存区到本地仓库|
|git reset|回退版本|
|git rm|删除工作区文件|
|git mv|移动或重命名工作区文件|
|git log|查看历史提交记录|
|git blame <file>|以列表形式查看指定文件的历史修改记录|
|git remote|远程仓库操作|
|git fetch|从远程获取代码库|
|git pull|下载远程代码并合并|
|git push|上传远程代码并合并|

## 上传github
git add .
git commit -m "注释"
git pull --rebase origin main
git push -u origin main

## 删除master分支
删除本地分支：git branch -d 分支名称

强制删除本地分支：git branch -D 分支名称

删除远程分支：git push origin --delete 分支名称

git config --global --unset http.proxy