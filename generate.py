#!/usr/bin/env python2.7.13
#coding: utf-8
import sys
import MySQLdb
from DBUtils.PooledDB import PooledDB
from multiprocessing.dummy import Pool

HOST = "127.0.0.1"
PORT = 3306
USER = "root"
PASSWORD = "afch5inattbc_13kl47gh"
DATABASE = "addata"
POOL_SIZE = 5

if len(sys.argv) !=3:
    print "python generate.py <platform> <day>"
    print "E.g. python generate.py <gdt|iqiyi|baidu|sina|mz> 20171213"
    sys.exit(0)

platform = sys.argv[1]
day = sys.argv[2]

names = {
        "iqiyi": "爱奇艺",
        "mz":"秒针",
        "gdt":"广点通",
        "sina":"新浪",
        "iqiyi_h5":"爱奇艺h5",
        "toutiao":"今日头条",
        "mobads":"百度SDK",
        }
class ConnMysql(object):
    pool = PooledDB(MySQLdb, POOL_SIZE, host=HOST, port=int(PORT), user=USER,
            passwd=PASSWORD, db=DATABASE, charset="utf8")

    def __init__(self):
        self.__conn = ConnMysql.pool.connection()
        self.__cursor = self.__conn.cursor()

    def select(self, query=''):
        try:
            self.effect = self.__cursor.execute(query)
            return self.__cursor
        except Exception as e:
            print(e)

    def update(self, query=''):
        try:
            self.effect = self.__cursor.execute(query)
            self.__conn.commit()
        except Exception as e:
            print(e)
            self.__conn.rollback()

    def __del__(self):
        self.__cursor.close()
        self.__conn.close()


# 连接MySQL，创建MySQL连接池。
mysql = ConnMysql()

sql = "select * from adtable where platform = '{}' and day='{}'; ".format(platform, day)
#print sql
res = mysql.select(sql)
#print res.fetchall()

#sys.exit()

fn = "iqiyi_tmp_out1.txt"
stat_map = {}
info_map = {}

"""
(604L, u'http://19.vrm.cn/40?src=gdt-kp-1106&qz_gdt=ia5dcwuf5meyp3sndcvq', 
u'\u4e2d\u56fd\u5e73\u5b89-\u5065\u5eb7\u662f\u6700\u957f\u60c5\u7684\u544a\u767d\u2014\u4e07\u4e08\u91d1\u6570DMP\u670d\u52a1', 
u'47f7cf9b65f3df800381decf2639cdc3.png', 1L, u'gdt', u'', u'', 20171213L)
"""
for row in res.fetchall():
    url = row[1]
    title = row[2].encode('utf-8')
    if row[3] is not None:
        img_file = "thumbnail/" +row[3]
    else:
        img_file = ""
    click = int(row[4])
    platform = row[5]
    day = row[8]
    stat_map[url] = stat_map.get(url,0) + click
    info_map[url] = (url, title, img_file, platform)


#with open(fn) as f:
#    for line in f:
#        flds = line.strip().split('\t')
#        if len(flds) >=6:
#            pv = int(flds[0])
#            url = flds[2]
#            title = flds[3]
#            img_file = flds[4]
#            platform = flds[5]
#            key = (url, title, img_file, platform)
#            stat_map[url] = stat_map.get(url,0) + pv 
#            info_map[url] = key

stat_map = sorted(stat_map.items(), key=lambda d: d[1], reverse=True) 

res = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{}</title>
    <meta content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=0" name="viewport"/>
    <meta content="yes" name="apple-mobile-web-app-capable"/>
    <meta content="black" name="apple-mobile-web-app-status-bar-style"/>
    <meta content="telephone=no" name="format-detection"/>
    <link rel="stylesheet" href="../css/ydui.css?rev=@@hash"/>
    <link rel="stylesheet" href="../css/demo.css"/>
    <script src="../js/ydui.flexible.js"></script>
</head>
<body>
<section class="g-flexview">

    <header class="m-navbar">
        <a href="../index.html" class="navbar-item"><i class="back-ico"></i></a>
        <div class="navbar-center"><span class="navbar-title">{}-{}</span></div>
    </header>

    <section class="g-scrollview">
        <article class="m-list list-theme3">
""".format(platform, names.get(platform,platform), day)


for idx, (url, stat) in enumerate(stat_map):
    url_short = url
    if len(url_short) > 80:
        url_short = url_short[:80]
    _, title, img_file, platform = info_map.get(url)
    res += "\n" 
    res += """
            <a href="{}" class="list-item" target='_blank'>
                <div class="list-img">
                    <img src="{}">
                </div>
                <div class="list-mes">
                            <span class="list-price"><em></em> No. {} &nbsp;</span>
                            <span class="list-del-price"><em>&nbsp;</em> {}</span>
                </div>
                <div class="list-mes">
                    <h3 class="list-title">
					        {}
					</h3>
                </div>
            </a>
        """.format(url,img_file,idx+1,stat, title)


res += """
       </article>
    </section>
</section>
<script src="http://apps.bdimg.com/libs/jquery/2.1.4/jquery.min.js"></script>
<script src="../js/ydui.js"></script>
<script>
    !function () {
        $('.m-list').find('img').lazyLoad({binder: '.g-scrollview'});
    }();
</script>
</body>
</html>
"""
    
with open("{}_{}.html".format(platform, day),"w") as f:
    f.write(res)

