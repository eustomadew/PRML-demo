# PRML 模式识别与机器学习
[![Documentation Status](https://readthedocs.org/projects/prml-demo/badge/?version=latest)](https://prml-demo.readthedocs.io/en/latest/?badge=latest) 

阅读文档 [PRML-demo](https://prml-demo.readthedocs.io/)



# ReadtheDocs 构建文档

[Demo Preparation](https://readthedocs-demo-zh.readthedocs.io/zh_CN/latest/%E6%96%87%E4%BB%B6%E6%89%98%E7%AE%A1%E7%B3%BB%E7%BB%9F-ReadtheDocs.html)

## QuickStart 本地

\# 1. 搭建环境
```shell
$ source ~/VirtualEnv/py36env/bin/activate
$ cd ~/Software
$ git clone https://github.com/sphinx-doc/sphinx
$ cd sphinx
$ pip install .
```
\# 2. 创建文档
```shell
$ cd ~/GitHubLab/ReadTheDocs
$ mkdir docs
$ cd docs
$ sphinx-quickstart # 生成docs项目

1. Separate source and build directories (y/n) [n]: y
2. Project name: helloworld
3. Author name(s): eustomaqua
4. Project release []: 0.1.1
5. Project language [en]: zh_CN

$ make html # 编译
$ # 预览页面: build\html\index.html
```
\# 3. 配置主题
```shell
$ pip install sphinx_rtd_theme
$ # source/conf.py 配置
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
$ make html
$ # 主题选择请查阅 https://sphinx-themes.org/
```
\# 4. 配置 markdown
```shell
$ pip install recommommark
$ # source/conf.py 配置
    增加：extensions = ['recommonmark'] 


$ # 增加 source/README.md 文档
***************************************
# helloworld

## 你好，Read the Docs
***************************************

    
4.修改source/index.rst，如下：
***************************************

    .. helloworld documentation master file, created by
       sphinx-quickstart on Sat Jun  1 13:29:49 2019.
       You can adapt this file completely to your liking, but it should at least
       contain the root `toctree` directive.
    
    Welcome to helloworld's documentation!
    ======================================
    
    .. toctree::
       :maxdepth: 2
       :caption: Contents:
        //增加配置，这行注释请去除
        README.md
    
    Indices and tables
    ==================
    
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
    
*************************************** 
```

## 关联 GitHub 与 ReadtheDocs

\# 1. 关联 GitHub

1. 在 GitHub 上创建 PRML-demo repo
2. 在本地创建 PRML-demo 文件夹，并把刚才的 docs 文件夹移入
3. 在 PRML-demo 文件夹内创建 .gitignore ，忽略 `docs/build/` 文件夹
4. 上传 GitHub

```shell
$ cd ~/GitHubLab/ReadTheDocs
$ mkdir PRML-demo
$ mv docs PRML-demo/
$ cd PRML-demo
$ vim .gitignore
docs/build/

$ git init
$ add .
$ git commit -m "first commit"
$ git remote add origin https://github.com/eustomaqua/PRML-demo.git
$ git push -u origin master
```

\# 2. 关联 Read the Docs

1. 打开网址 [https://readthedocs.org/](https://readthedocs.org/)
2. 点击菜单：`我的项目`
3. 点击按钮：`Import a Project`
  - 若导入失败，尝试 `刷新您的账号`
  - 或右侧 `手动导入`，填入项目名称 PRML-demo 和代码库地址，代码库类型默认 Git 不变
4. 构建项目：`Build version`
  - 若报错 contents.rst not found
  - 修改后点击项目上方 `构建`
5. 构建成功后，点击 `阅读文档`

若报错 contents.rst not found 
```shell
打开配置source/conf.py，增加如下配置：

# The master toctree document.
master_doc = 'index'

产生这个问题的原因：默认首页名称是contents.rst；
更改一下就可以了，重新提交github，提交后，点击“构建”按钮，等待Read the Docs自动构建；
```

## 多语言文档

参考 [文档本地化](https://readthedocs-demo-zh.readthedocs.io/zh_CN/latest/%E6%96%87%E4%BB%B6%E6%89%98%E7%AE%A1%E7%B3%BB%E7%BB%9F-ReadtheDocs.html#id5)

```shell
$ cd ~/GitHubLab/ReadTheDocs
$ cd PRML-demo/docs
$ mkdir zh_CN
$ mkdir en
# 把开始 docs 下的文件都放入 zh_CN
```

构建 en 版本的 Read the Docs
```shell
$ cd PRML-demo/docs/en
$ sphinx-quickstart

执行以上命令会弹出命令框：
// build与source是否隔开，build是存放编译后的文件，source是放配置文件
1. Separate source and build directories (y/n) [n]: y 
// 项目名称
2. Project name: helloworld
// 作者
3. Author name(s): eustomaqua
// 项目版本
4.  Project release []: 0.1.1
// 语言，英语：en
5. Project language [en]: en

# 修改 docs/en/source/conf.py
# 增加 docs/en/source/README.md ，修改 docs/en/source/index.rst
```

项目默认关联 en 英文，修改成双语
1. 在项目中点击 `管理`->`设置`，修改语言 `Simplified Chinese`，保存后等待重新构建完成
2. 在项目中点击 `管理`->`高级设置`，修改`Python 配置文件`项，docs/zh_CN/source/conf.py，点击保存，等待构建完成
