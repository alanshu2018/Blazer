# coding: utf-8
import logging
import threading
import random
import redis
import base64
import sys
import MySQLdb
from config import *
from crawl_base import *
from ad_filter import *
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s [%(levelname)s]:%(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S' )
# g_logger = logging.getLogger()
from redis_bloom import *
reload(sys)
sys.setdefaultencoding('utf-8')


if DEBUG or ONLY_GDT:
    redis_pop_num = 1
    num_threads = 1
elif DEBUG_CRAWL:
    redis_pop_num = 1
    num_threads = 1

crawl_qname_list =[]
if len(sys.argv) >=2 and sys.argv[1] == "crawl_error":
    crawl_qname_list = ["crawl_error_data"]
    do_error_crawl = True
    num_threads = 128
    redis_pop_num = 32

class CaptureDB(object):
    def __init__(self):
        pass

    def open_db(self):
        return MySQLdb.connect(
                host=db_config['host'],
                user=db_config['user'],
                passwd=db_config['passwd'],
                db=db_config['db'],
                port=db_config['port'],
                charset='utf8'
        )
    def close_db(self,db):
        db.close()

    def fetch_rows(self,sql):
        db = self.open_db()
        cursor = db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            rows = []
            for row in result:
                rows.append(row)
            return rows
        except Exception, e:
            g_logger.info("DB Fetch rows error:{},sql={}".format(e,sql))
        finally:
            cursor.close()
            db.close()
        return None

    def execute(self,sql):
        db = self.open_db()
        cursor = db.cursor()
        try:
            cursor.execute(sql)
            return True
        except Exception, e:
            #g_logger.info("DB Execute error:{}, sql={}".format(e,sql))
            g_logger.error("db execute error!!!")
        finally:
            cursor.close()
            db.commit()
            db.close()

        return False


class CrawlThread(threading.Thread):
    invalid_suffix_set = set([
        ".jpg",".jpeg",".png", ".gif",".ico",
        ".css",".js",".mp4",".zip",".mp3",".m3u8",".ashx",".swf"
    ])
    invalid_url_string = [
        ".ykimg.com",
        "/image.",
        ".qiyipic.com",
        ".iqiyi.com/lib/",
        "pic.",
        "pic1.",
        "pic2.",
        "pic3.",
        "pic4.",
        "pic5.",
        "tv.cctv.com",
        "m.le.com",
        "css.",
        "/css",
        "js.",
        "/js/",
        "/image/",
        "tags.",
        "logs.",
        "tjhm.",
        ".lecloud.com",
        "/initJs",
        "format=jpg",
        "Config.",
        "Config?",
        "img01.",
        "img02.",
        "img03.",
        "img04.",
        "img.",
        "img1.",
        "img2.",
        "img3.",
        "img4.",
        "image.",
        "jssdk.",
        "dev.",
        "=json",
        "wap.yanru.net",
        "article",
        "/font/",
        "tracking.",
        "interface.",
        "votes",
        "callback=",
        "media",
        "comicDetail",
        "h5.fbianli.com",
        "personalRecommend",
        "video",
        "open.weixin.qq.com",
        "d5.mobaders.com",
        "www.baidu.com",
        "lezhibo.tianqist.cn",
        "v2html.atm.",
        "www.giorgioarmanibeauty.cn",
        "youku"
    ]

    configs = [data_redis,redis_config2, redis_config3]
    redis_handles = [redis.Redis(host=config['host'], port=config['port'], db=config['db']) for config in configs]

    normal_redis = redis.Redis(host=data_redis['host'], port=data_redis['port'], db=data_redis['db'])
    #main_redis = redis.Redis(host=redis_config1['host'], port=redis_config1['port'], db=redis_config1['db'])
    test_redis = redis.Redis(host=redis_config_test['host'], port=redis_config_test['port'], db=redis_config_test['db'])

    bloom_filter = BloomFilter(data_redis)

    #add redis
    adsee_redis=redis.Redis(host=adsee_redis['host'],port=adsee_redis['port'],db=adsee_redis['db'])
    crawl_error_redis=redis.Redis(host=crawl_error_redis['host'],port=crawl_error_redis['port'],db=crawl_error_redis['db'])
    notitle_or_nopic_redis=redis.Redis(host=notitle_or_nopic_redis['host'],port=notitle_or_nopic_redis['port'],db=notitle_or_nopic_redis['db'])
    landing_url_redis=redis.Redis(host=landing_url_redis['host'],port=landing_url_redis['port'],db=landing_url_redis['db'])

    #ad_url_redis=redis.Redis(host=ad_url_redis['host'],port=ad_url_redis['port'],db=ad_url_redis['db'])
    #end
    main_redis = redis.Redis(host=data_redis['host'], port=data_redis['port'], db=data_redis['db'])

    day_map = {}
    g_db = CaptureDB()

    toutiao_crawler = ToutiaoAdCrawler()
    main_crawler = GoldenCrawler()

    def __init__(self, idx, test_redis_mode=False):
        threading.Thread.__init__(self)
        self.tid = idx
        self.test_redis_mode = test_redis_mode
        if self.test_redis_mode:
            self.main_redis = self.test_redis
            self.redis_handles = [self.test_redis]
            self.configs = [redis_config_test]
        else:
            self.main_redis = self.normal_redis

    def g_info(self, msg):
        g_logger.info("{:0>2d}, {}".format(self.tid, msg))

    def g_error(self, msg):
        traceback.print_exc()
        g_logger.error("{:0>2d}, {}".format(self.tid, msg))

    def get_suffix(self, url):
        url = get_suffix(url)
        return url

    def check_crawl_queue(self, r, name):
        try:
            data_list = r.lrange(name, 0, 0)
            data = data_list[0]
            data = json.loads(data)
            if data.has_key('platform') and data.has_key('os') and data.has_key('area'):
                return True
        except:
            pass
        return False

    def save_data_db(self,data):
        #g_logger.info ("Going to save_data_db={}".format(data))
        title = data['title']
        pic = data['pic']
        other_conds = []
        if len(pic) > 0:
            other_conds.append("pic='{}'".format(pic))

        if len(title) > 0 and title !='no_title':
            other_conds.append("title='{}'".format(title))

        other_stat = ""
        if len(other_conds) > 0:
            other_stat = ','.join(other_conds)

        sql = "INSERT INTO " + db_config['table'] \
              + """ 
                (landing_url,title, thumbnail, click, platform, advertiser, category, day, ourl, ad_url, pic, area, os, ua)
                VALUES (
                "{landing_url}","{title}","{thumbnail}",{click},"{platform}",
                "{advertiser}","{category}","{day}","{ourl}","{url}","{pic}",
                "{area}","{os}","{ua}") 
                on duplicate key update click=click+{click}""".format(**data)

        if self.g_db.execute(sql):
            #self.g_info ("OK to save_data_db={}\nsql={}".format(data,sql))
            self.g_info("ok to save db,host={},area={},day={}".format(get_host(data['url']),data['area'],data['day']))

        if len(other_stat) > 0:
            update_sql = "UPDATE " + db_config['table'] + " SET " + other_stat \
                + " WHERE landing_url='{}' AND day='{}' ".format(data['landing_url'], data['day'])

            if self.g_db.execute(update_sql):
                self.g_info("ok to update db,host={},area={},day={}".format(get_host(data['url']), data['area'], data['day']))
                #sql += other_stat

    # Increase counter in data
    def increase_data_db(self, data, click):
        #g_logger.info ("Going to save_data_db={}".format(data))
        landing_url = data['landing_url']
        day = data['day']
        area = data['area']
        platform = data['platform']

        update_sql = "UPDATE " + db_config['table'] + """
                    SET click= click +{}
                    WHERE landing_url='{}' AND day='{}' and platform='{}' and area='{}'
                """.format(click, landing_url, day, platform, area)

        if self.g_db.execute(update_sql):
            self.g_info("ok to save db,host={},area={},day={}".format(get_host(data['url']),data['area'],data['day']))
            #sql += other_stat

    def extract_raw_key(self, url):
        url, rw_url = extract_raw_key(url)
        return url, rw_url

    def merge_items(self,items):
        url_item_map = {}
		today=int(get_day(0))
        for item_json in items:
            try:
                item = json.loads(item_json)
                url = item['url']

				dday = int(get_day(item['time']))
				if today - dday > 1:
					continue
                #if not is_ad_url(url): # is_ad_url only works for iqiyi
                #    self.g_info("ok to filter data from redis,host={},area={},day={}".format(get_host(url),item['area'],get_day(item['time'])))
                #    #if url.find("tracking.v.tf.360.cn")>=0 or url.find("lezhibo.tianqist.cn")>=0:
                #    #self.g_info("ok to filter data from redis,host={},area={},day={}".format(get_host(url),item['area'],get_day(item['time'])))
                #    continue
                # Filter out invalid url
                suffix = self.get_suffix(url)
                if suffix in self.invalid_suffix_set:
                    #g_logger.info("filter url={}",url)
                    self.g_info("ok to filter data from read,host={},area={},day={}".format(get_host(url),item['area'],get_day(item['time'])))
                    continue
                b_invalid_contained = False
                for u in self.invalid_url_string:
                    if url.find(u) >=0:
                        #g_logger.info("filter url={}", url)
                        self.g_info("ok to filter data from read,host={},area={},day={}".format(get_host(url), item['area'], get_day(item['time'])))
                        b_invalid_contained = True
                        break
                if b_invalid_contained:
                    continue

                url, rw_url = self.extract_raw_key(url)
                if len(url) < 4:
                    self.g_info("ok to filter data from read,host={},area={},day={}".format(get_host(url), item['area'],
                                                                                            get_day(item['time'])))
                    continue

                # Rewriting url if necessary
                if len(rw_url) > 0:
                    item['url'] = rw_url

                try:
                    item['ukey'] = url.encode("utf-8")
                except:
                    item['ukey'] = url

                platform = item['platform']
                if platform == 'iqiyi':
                    if url.find("www.iqiyi.com") >= 0 \
                            or url.find('mgtv') > 0 \
                            or url.find('/api/news/') > 0 \
                            or url.find('i.iqiyi.com') > 0 \
                            or url.find('tv.com') > 0 \
                            or url.find('piao.iqiyi.com') >= 0 \
                            or url.find('server.iqiyi.com') >= 0 \
                            or url.find('zhaopin.iqiyi.com') >= 0 \
                            or url.find('toutiao.iqiyi.com') >=0 \
                            or url.find('m.iqiyi.com') >= 0:
                        continue
                key = url + "\t" + platform
                data = url_item_map.get(key, None)
                if data is None:
                    url_item_map[key] = item
                else:
                    url_item_map[key]['pv'] += item['pv']
            except Exception, e:
                self.g_error("merge_items exception={}".format(e))
                continue

        items = []
        for key, item in url_item_map.items():
            item['day'] = get_day(item['time'])
            items.append(item)

        return items

    def check_cached_crawl_list(self, ukey_day_list, day_map):
        total_keys = 0
        total_hit = 0
        day_map1 = {}
        for ukey_day in ukey_day_list:
            ukey, day = ukey_day
            map = day_map1.get(day,{})
            map[ukey] = None
            day_map1[day] = map
            total_keys += 1

        if len(day_map1) <= 0:
            return day_map

        try:
            # 本地
            for day, map in day_map1.items():
                ukey_list = []
                for ukey,_ in map.items():
                    ukey_list.append(ukey)
                results = self.main_redis.hmget("cache"+day,[get_md5(ukey) for ukey in ukey_list])
                for ukey, ret in zip(ukey_list,results):
                    if ret is not None and len(ret) > 0:
                        try:
                            total_hit += 1
                            if day_map.get(day,None) is None:
                                day_map[day] = {}
                            day_map[day][ukey] = json.loads(ret)
                        except Exception,e:
                            self.g_error("check_cached_crawl_list,json parse error:json={},exception={}" \
                                    .format(ret, e))
        except Exception, e:
            traceback.print_exc()
            self.g_error("check_cached_crawl_list:exception={}".format(e))

        self.g_info("check_cached_crawl_list: total={},hit={}, ratio={}%" \
                      .format(total_keys,total_hit,total_hit*100.0/total_keys))
        return day_map

    def save_stat_to_redis(self, day, platform, key, cnt):
        try:
            # 本地
            self.main_redis.hincrby('hstat_'+day,platform + key ,cnt)
        except Exception, e:
            self.g_error("save_pic_to_redis:exception={}".format(e))

    """
    Crawl
    """
    def crawl_url(self, data):
        # data = json.loads(data)
        time = data["time"]
        ua = data["ua"]
        #ostype = data["os"]
        ref = data.get("referer", "")
        url = data["url"]  # original url
        #sorted_url = data["landing_url"]  # check key
        click = data["pv"]
        platform = data["platform"]
        area= data.get("area","hnlt")
        ostype=get_os(ua)
        day= data.get('day',get_day(time))
        tries = data.get('tries',0)

        if tries == 0:
            self.save_stat_to_redis(day,platform, "recv", click)
        # try:
        #     self.g_info("crawling_data:url={}".format(url))
        # except Exception,e:
        #     traceback.print_exc()
            #print(data)

        if url[:7] != "http://" and url[:8] != "https://" :
            url = "http://" + url

        """
        #由于adsee redis 中的数据一部分来源于crawler ,另一部分来自于adsee,
        一个url是完整的,另一个是截断后的.所以在此处,再查找一下redis.
        """
        tmp_url,rw_url = self.extract_raw_key(url)
        url_key = get_md5(tmp_url)
        title,pic,landing_url = self.get_adsee_data(url_key)
        landing_url_new, rw_url = self.extract_raw_key(landing_url)

        if title!="" and pic!="" and landing_url!="":
            data = {
                'ourl': landing_url,  # final url
                'url': url,  # original key, like ad_url
                'landing_url': landing_url_new,  # sorted_url, #set to sorted key
                # 'http://ysa.xiaoliwen.com/ysaqyx6/?td_ref=http%3A%2F%2Flnk0.com%2FwpIpoc',
                'pic': pic,
                'title': title,
                'thumbnail': "",
                'click': click,
                'platform': platform,
                'advertiser': '',
                'category': '',
                'day': day,
                'area': area,
                'os': ostype,
                'ua': ua,
            }
            return data

        #apk处理start
        if url.find('.apk') > 0:
            title = get_apk_title(url)
            img_url = 'uimages/apkapkapk.jpg'
            landing_url = url
        #######end
        else:

            if url.find('ad.toutiao.com/tetris/page')>0:
                #self.g_info("toutiao--------------page!!!!!")
                title, img_url, landing_url = self.toutiao_crawler.crawl(url, ref, ua, ostype, platform,area)
                notitle, img_url, nolanding_url = self.main_crawler.crawl(url, ref, ua, ostype, platform, area)
            else:
                title, img_url, landing_url = self.main_crawler.crawl(url, ref, ua, ostype, platform,area)

            if landing_url == "" \
                    or title.find('404') >= 0 \
                    or landing_url.find('.189.cn/') >= 0 \
                    or not is_ad_url(landing_url):
                #self.g_error("Crawl Error and save into redis: url=%s, landing_url=%s, title=%s" % (url, landing_url, 'title'))
                if not is_ad_url(landing_url) or landing_url.find('.189.cn/') > 0:
                    pass
                else:
                    self.g_error("fail to crawl,host={},area={},day={}".format(get_host(url), area, day))
                data['tries'] = tries + 1
                if tries < 2:
                    self.save_error_to_redis(data)
                return

            # if landing_url!="" and (title!="no_title" or img_url==''):
            #     #adsee的key可能是落地页链接,用图片为空或者title为空的落地页去尝试!!!
            #     landing_url_key = get_md5(landing_url)
            #     tmp_title, tmp_pic, useless_url = self.get_adsee_data(landing_url_key)
            #     if img_url=="" and tmp_pic!="":
            #         img_url = tmp_pic
            #     if title=="" and tmp_title!="":
            #         title = tmp_title

            if title == "":
                title = "no_title"
                apk_title = get_apk_title(landing_url)
                if len(apk_title) > 0:
                    title = apk_title
                    img_url = 'uimages/apkapkapk.jpg'

            if title == "no_title" or img_url == "":
                self.save_no_picortitle_to_redis(url, landing_url, title, img_url, ua, ref)
            else:
                self.save_finally_data_to_adsee_redis(url,landing_url,title,img_url)
        #save stat
        self.save_stat_to_redis(day, platform, "ok", click)

        landing_url_new, rw_url = self.extract_raw_key(landing_url)

        data = {
            'ourl': landing_url,  # final url
            'url': url,  # original key, like ad_url
            'landing_url': landing_url_new,  # sorted_url, #set to sorted key
            # 'http://ysa.xiaoliwen.com/ysaqyx6/?td_ref=http%3A%2F%2Flnk0.com%2FwpIpoc',
            'pic': img_url,
            'title': title,
            'thumbnail': "",
            'click': click,
            'platform': platform,
            'advertiser': '',
            'category': '',
            'day': day,
            'area': area,
            'os':ostype,
            'ua':ua,
        }
        return data

    def merge_and_crawl(self, items):
        if len(items) <=0:
            return

        merged = self.merge_items(items)
        self.g_info("Merged items.cnt=%d" % (len(merged)))

        ##########################################################
        # Filter out the crawl error page
        ##########################################################
        if len(merged) <= 0:
            return

        filtered = []
        for item in merged:
            is_bad = False
            try:
                my_url = item['url']
                if my_url[:4] != "http":
                    my_url = "http://" + my_url
                    item['url'] = my_url

                #self.g_info("Filter check url: url=%s" % (my_url))
                if self.bloom_filter.isContains(my_url):
                    is_bad = True
                    self.g_info("Filter bad url: url=%s" % (my_url))
            except:
                pass
            finally:
                if is_bad == False:
                    filtered.append(item)

        self.g_info("Filter items: total=%d, keep=%d, keep_ratio=%.2f%%" % (len(merged), len(filtered), len(filtered)*100.0/len(merged)))

        merged = filtered
        ##########################################################
        #  check cached result to avoid unnecessary crawling
        ##########################################################
        ukey_day_list = []
        for item in merged:
            day=item['day']
            ukey = item['ukey']

            if len(ukey) < 1:
                #self.g_error("===>Not set ukey for item:{}".format(item))
                continue

            ukey_day_list.append((ukey,day))

        # save items for crawl
        to_crawl_items = []
        day_map = {}
        self.check_cached_crawl_list(ukey_day_list,day_map)
        self.g_error("===>day_map:num={}".format(len(day_map)))
        total_cached = 0
        for item in merged:
            day=item['day']
            ukey = item['ukey']
            ret = day_map.get(day,{}).get(ukey,None)
            if ret is None:
                to_crawl_items.append(item)
            else:
                total_cached += 1
                click = item.get('pv',item.get('click',1))
                self.increase_data_db(ret,click)

        day_map.clear()
        self.g_error("===>day_map cached items:num={}".format(total_cached))

        # reset
        ukey_day_list = []
        ##########################################################
        #  Do real crawl for the items
        ##########################################################
        ukey_day_data_list = []
        for item in to_crawl_items:
            try:
            # crawl
                ukey = item['ukey']
                day = item['day']
                data = self.crawl_url(item)
                # save into db
                if data is not None:
                    landing_url = data.get('landing_url','')
                    if landing_url.find('.baidu.com/') >0 \
                            or landing_url.find('zt.jd.com/ad/appjump') >0 \
                            or landing_url.find('.gdt.qq.com/gdt_mclick.fcg') >0:
                        # Remove error pages
                        continue

                    self.save_data_db(data)
                    data['day'] = day
                    ukey_day_data_list.append((ukey,day,data))
                    # save image into redis
                    if data.get("pic") != '':
                        self.save_pic_to_redis(data['pic'])
                    time.sleep(2)
            except Exception, e:
                traceback.print_exc()
                self.g_error("===>Exception:{}".format(e))
            finally:
                pass
            if len(ukey_day_data_list) > 0:
                self.save_cached_crawl_list(ukey_day_data_list)
                ukey_day_data_list = []

    def fetch_queue_items(self, queues_list):
        batch_num = 50
        gdt_ration = 1000

        while to_exit == False:
            for config, r in zip(self.configs,self.redis_handles):
                time.sleep(2)
                self.g_info("="*80)
                self.g_info("Read redis using config={}".format(config))
                my_queues = []
                if len(queues_list) > 0:
                    my_queues = queues_list
                else:
                    try:
                        my_queues = []
                        for queue in r.keys():
                            if self.check_crawl_queue(r, queue):
                                my_queues.append(queue)

                        # remove crawl_error
                        my_queues = [ qname for qname in my_queues if qname !="crawl_error_data"]

                    except Exception, e:
                        self.g_error("Get Quene names error:{}".format(e))

                random.shuffle(my_queues)

                self.g_info("Queue names={}".format(','.join(my_queues)))

                total_per_config = 0
                for qname in my_queues:
                    try:
                        total_length = r.llen(qname)
                        self.g_info("Queue {} has {} items in total".format(qname, total_length))
                        num = 0
                        # pop batch_num items
                        while num < batch_num and total_per_config < 5000:
                            #item = r.lpop(qname)
                            item = r.rpop(qname)
                            if qname == "gdt":
                                # only use 1/10 of gdt items
                                if random.randint(0,1000) >= gdt_ration:
                                    continue

                            if item is not None:
                                num += 1
                                total_per_config += 1
                                yield item

                            if r.llen(qname) <= 0:
                                break
                    except Exception,e:
                        self.g_error("Fetch_queue_items exception:{}".format(e))
                    finally:
                        time.sleep(1)

    def save_pic_to_redis(self,img_path):
        try:
            if img_path=="uimages/apkapkapk.jpg":
                return
            if not os.path.exists(img_path):
                return
            # 本地
            with open(img_path, 'rb') as fin:
                bs4_str = base64.b64encode(fin.read())
                img_data = {
                    'img_path': img_path,
                    'img_bs4_str': bs4_str
                }
                img_data = json.dumps(img_data)
                self.main_redis.lpush("image", img_data)
                #add remove imgfile after lpush
                self.g_info("ok save picture to redis,path={}".format(img_path))
                os.remove(img_path)
                self.g_info("ok to del local picture,path={}".format(img_path))
        except Exception, e:
            self.g_error("save_pic_to_redis:exception={}".format(e))

    def save_error_to_redis(self, data):
        data_json = json.dumps(data)
        if do_error_crawl:
            # save error into redis to avoid try many times
            try:
                url = data['url']
                self.bloom_filter.insert(url)
                self.g_error("save_error_crawl_to_bloomfilter:data={}".format(url))

                #add
                self.crawl_error_redis.set(get_md5(url),True)
                #
            except Exception, e:
                self.g_error("save_error_crawl_to_redis:data={},exception={}".format(data_json,e))
            return
        try:
            self.main_redis.lpush("crawl_error_data",data_json)
        except Exception, e:
            self.g_error("save_error_crawl_to_redis:data={},exception={}".format(data_json,e))

    def save_no_picortitle_to_redis(self, url, landing_url, title, pic,ua,ref):
        #adurlkey = get_md5(url)
        #tmp_url,rw_url = self.extract_raw_key(url)
        tmp_landing_url,rw_url = self.extract_raw_key(landing_url)
        info = {'adurl': url, 'landing_url': tmp_landing_url, 'title': title, 'pic': pic,'ua':ua,'ref':ref}
        sinfo = json.dumps(info)
        self.notitle_or_nopic_redis.rpush("crawler_fail", sinfo)
        #logging.info("ok save no pic or no title,data={%s}",info)

    def save_finally_data_to_adsee_redis(self,url, landing_url, title, img_url):
        landing_url_new, rw_url = self.extract_raw_key(landing_url)
        url_new, rw_url = self.extract_raw_key(url)
        info = {'ad_url':url,'landing_url':landing_url_new,'title':title,'pic':img_url}
        adurlkey = get_md5(url_new)
        landing_urlkey = get_md5(landing_url_new)
        sinfo = json.dumps(info)
        #self.adsee_redis.mset({adurlkey:sinfo,landing_urlkey:sinfo})
        self.adsee_redis.set(adurlkey,sinfo)
        self.landing_url_redis.set(landing_urlkey,sinfo)
        #logging.info("ok save pic and title to adsee,title={%s},pic={%s}",title,img_url)

    def save_cached_crawl_list(self, ukey_day_data_list):
        day_map = {}
        for ukey_day_data in ukey_day_data_list:
            ukey, day, data = ukey_day_data
            map = day_map.get(day,{})
            map[get_md5(ukey)] = json.dumps(data)
            day_map[day] = map

        try:
            # 本地
            t = int(time.time()) + 600
            for day, map in day_map.items():
                self.main_redis.hmset("cache"+day,map)
                #prevent redis to eat too much memory
                self.main_redis.expireat("cache"+day,t)
        except Exception, e:
            traceback.print_exc()
            g_logger.error("set_cached_crawl:exception={}".format(e))
        finally:
            day_map.clear()

        return None

    def find_and_save_from_adsee(self,json_data):
        flag = False
        try:
            data = json.loads(json_data)
            url = data['url']
            adurlkey = get_md5(url)
            title, pic ,landing_url = self.get_adsee_data(adurlkey)
            if title !="" and pic!='':
                ua = data["ua"]
                time = data["time"]
                ref = data.get("referer", "")
                click = data["pv"]
                platform = data["platform"]
                area = data.get("area", "hnlt")
                ostype = get_os(ua)
                day = data.get('day', int(get_day(time)))

                #是本地路径
                # if landing_url == "":
                #     landing_url = "url"

                logging.info("***********"*10)
                logging.info("ok find data from adseedata or crawler,title={},pic={}".format(title,pic))
                logging.info("***********"*10)

                save_data={
                    'ourl':url,  # final url
                    'url': url,  # original key, like ad_url
                    'landing_url': landing_url,  # sorted_url, #set to sorted key
                    'pic': pic,
                    'title': title.encode('utf-8'),
                    'thumbnail': "",
                    'click': int(click),
                    'platform': platform,
                    'advertiser': '',
                    'category': '',
                    'day': int(day),
                    'area': area,
                    'os': ostype,
                    'ua': ua,
                }

                #pic应该已经存在,不再传图片
                #self.save_pic_to_redis(pic)

                self.save_data_db(save_data)
                g_logger.info("ok save to db from adseedata or crawler!!!!!")
                flag = True
        except Exception,e:
            traceback.print_exc()
            print e
        finally:
            return flag

    def get_adsee_data(self,adurlkey):
        json_data = self.adsee_redis.get(adurlkey)
        title = ''
        pic = ''
        landing_url = ''
        if json_data is not None:
            data=json.loads(json_data)
            title=data['title']
            pic = data['pic']
            if data.has_key('landing_url'):
                landing_url = data['landing_url']
            else:
                self.adsee_redis.delete(adurlkey)
            if title != "" and pic != "" and landing_url!="":
                if pic.find('uimages')>=0:
                    logging.debug('read landing_url={} from adseedata in adsee_redis success !!!!'.format(landing_url))
                else:
                    pic = self.main_crawler.crawler.download_image("",img_path,pic,"","","","")
                    if pic!="":
                        self.save_pic_to_redis(pic)
                        logging.debug('read landing_url={} from previous crawler in adsee_redis success !!!!'.format(landing_url))
        return title, pic,landing_url

    def exist_crawl_error_redis(self,json_data):
        flag = False
        data = json.loads(json_data)
        url= data['url']
        url_key = get_md5(url)
        try:
            info = self.crawl_error_redis.get(url_key)
            if info:
                logging.info("ok find crawl error redis,url={}".format(url))
                flag = True
        except Exception,e:
            traceback.print_exc(e)
        return flag

    def run(self):
        self.g_info ("Crawl Thread %d enters"%(self.tid))
        total_items = 0
        while to_exit == False and total_items < 20000:
            items = []
            if DEBUG or ONLY_GDT:
                qnames = ['iqiyi']
            elif len(crawl_qname_list) > 0:
                qnames = crawl_qname_list
            else:
                qnames = [] #read_redis_key()

            last_update = time.time()
            for item in self.fetch_queue_items(qnames):
                try:
                    if item is None:
                        continue
                    #if item.find("pstatp.com") >=0 or item.find("snssdk.com") >=0:
                    #    continue
                    if not self.find_and_save_from_adsee(item) and not self.exist_crawl_error_redis(item):
                        items.append(item)
                        total_items += 1

                    if len(items) > redis_pop_num or time.time() - last_update > 120:
                        last_update = time.time()
                        self.merge_and_crawl(items)
                        items = []
                except Exception, e:
                    traceback.print_exc()
                    self.g_error("Crawl Exception:{}".format(e))

            if len(items) > 0:
                self.merge_and_crawl(items)

            time.sleep(1)
        self.g_info("Crawl Thread %d leaves"%(self.tid))

def t_crawl_redis():
    #start crawl thread
    threads = []
    for i in range(num_threads):
        t1 = CrawlThread(i, test_redis_mode=True)
        threads.append(t1)
        t1.start()

    time.sleep(10)

    for t in threads:
        t.join()

def t_crawl_url(url, platform):
    data = {}
    data["time"] = time.time()
    #data["ua"] = 'GDTADNetClient-[Dalvik/1.6.0 (Linux; U; Android 4.4.4; Che1-CL10 Build/Che1-CL10)]'
    data["ua"] = 'Mozilla/5.0 (Linux; Android 7.1.2; vivo Y66i Build/N2G47H; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/55.0.28'
    data["os"] = 'android'
    data["url"] = url #'http://app.qq.com/detail/100825615?channelid=000116083637343632323233&autodownload=0&qz_gdt=fkffowqbsqap5vdzvvua' #original url
    #data["landing_url"] = 'http://app.qq.com/detail/100825615?channelid=000116083637343632323233&autodownload=0&qz_gdt=fkffowqbsqap5vdzvvua'  # check key
    data["pv"] = 1
    data["platform"] = platform
    data["area"] = 'hubei-ct'

    t = CrawlThread(0)
    t.merge_and_crawl([json.dumps(data)])
    """
    data = t.crawl_url(data)
    if data is not None:
        landing_url = data.get('landing_url','')
        if landing_url.find('.baidu.com/') >0 \
                or landing_url.find('zt.jd.com/ad/appjump') >0 \
                or landing_url.find('.gdt.qq.com/gdt_mclick.fcg') >0:
            # Remove error pages
            return

        t.save_data_db(data)
        # save image into redis
        if data.get("pic") != '':
            t.save_pic_to_redis(data['pic'])
    """
if __name__ == "__main__":
    if DEBUG_REDIS:
        t_crawl_redis()
        sys.exit(0)
    elif DEBUG:
        url = "http://c.gdt.qq.com/gdt_mclick.fcg?viewid=M2LsS2eRU__yUlKzCNN4pbl7I_ao5JhYg_JeZCSlX_nRiPx5g_Qt7X!SpSYUrwKsLmvF0m_mP_g458DgnzTshxIJJIKv_bX13FR!MqtyBs2DlKAcjJG0QbxrEm80KWabAyKH6X5pwu2NNXwwtwwIjLWnaOJhWKjGATNC1l!kyFVKgxbB84lSVosrlixgvCSUcYmJzC861872WRPNe3kNls404AXS23mO&jtype=0&i=1&os=2&asi=%7B%22mf%22%3A%22OPPO%22%2C%22hw%22%3A%22mt6755%22%2C%22br%22%3A%22OPPO%22%7D&subordinate_product_id=67610070;000116083637353435313834&acttype="
        #url = "http://app.qq.com/detail/1101072624?channelid=000116083637363432343736&autodownload=0&qz_gdt=h2jvowtsrmakrit5vykq"
        url = "http://s.x.cn.xtgreat.com/ax?l=192298&r=1&c=window.__mz_collect_adx&v=2&f=&u=http%3A%2F%2Fm.80ml.com%2Fchaoliuqushi%2F2015%2F1118%2F54891.html&mv=j1."
        url = "http://static.iqiyi.com/ext/cupid/lp/59488af233260a7e624f15fc_1010000007651.html"
        url = "http://app.qq.com/detail/1105380575?channelid=000116083534363836353335&autodownload=0&qz_gdt=ycffowrrwiaiil5roltq"
        url = "http://www.ibmcpn.cn/url/388?t=151598150898"
        url = "http://ad.toutiao.com/tetris/page/1589907012457480/"
        url = "http://ad.toutiao.com/tetris/page/1587848017953805/?ad_id=83740708700&cid=83741282675&req_id=20180129143942010012022216889307"
        #url = "http://www.ibmcpn.cn/url/413?t=151598711625"
        url = "saxn.sina.com.cn/dsp/click?t=MjAxOC0wMS0yOCAyMzo0MToyOS4wNzYJNjEuMTU4LjE0OC4xMjAJX182MS41My4xMTEuODRfMTQ4NjcyMDMyMl8wLjY0NjA5OTAwCWIyYmUzYTk3LWJhOTctNDYzYy05OThlLTgxNWViNGM2N2E2ZAkzMTcyMjEyCTU4Mjc5MjM3NTFfUElOUEFJLUNQQwkzMDc3MDU5CTkxMDAwCTAuMDA5NjQ1MjQyCTEJdHJ1ZQlQRFBTMDAwMDAwMDU2NDMxCTMzODA5MjQJV0FQCXh4bAktCTF8N2VON3BQY29YYWNaN1Z5Y2dPQzFldXwzNXxudWxsfGJqfDF0Tkc1dUZMcUVTaHR2MERRNk44Y0l8MWhoVG95TUpndFI2YWxrNlliY1pBa3wwCW51bGwJMQktCS0JLQkwCV9fNjEuNTMuMTExLjg0XzE0ODY3MjAzMjJfMC42NDYwOTkwMAlXQVBfRkVFRAktCW5vcm1hbHx1dmZtLXJ0CS0JdXNlcl90YWc6MjEzNzg6MC4wMzR8d2FwX29zOjcwMTowLjB8dXNlcl9hZ2U6NjAxOjAuMjE4OTl8dXNlcl9nZW5kZXI6NTAxOjEuMHx2X3pvbmU6MzEyMDAxOjAuMHxuZXRfd2lmaToxMTAyOjAuMHxjcm93ZHM6fF9jcm93ZHM6CTIJOTEwMDAJNTAwMDAJLQ==&userid=__61.53.111.84_1486720322_0.64609900&auth=872be9dd4d20d1ca&type=0&c=0&url=http%3A%2F%2Fsax.sina.cn%2Fclick%3Ftype%3D2%26t%3DNzRmNjg1YmItOTcwNy0zNGYyLTlhZTUtNTkwOTkxYWQ1YjQ2CTQ1CVBEUFMwMDAwMDAwNTY0MzEJMzM4MDkyNAkxCVJUQgktCQk%253D%26id%3D45%26url%3Dhttp%253A%252F%252Fdigitaltobacco.org%252FF-XK%252FH1%252F%26sina_sign%3Dddfb044eda059f0c&sign=4cacbebac2e53bb4&p=dPaFu5cHNPKa5VkJka1bRsGVzr4JDxEai8Ekiw%3D%3D&cre=tianyi&mod=wmil&vt=4&pos=24"
        url = "http://pub.m.iqiyi.com/jp/h5/recommend/videos/?area=h_swan&size=60&type=video&tvid=116427500&referenceId=116427500&albumId=116427500&cookieId=0&channelId=7&trimUser=false&qyid=b3d649e5339ea42f5aa823ff8fa5ff30&_=1517933085515&callback=Zepto1517933082194"
        url = "www.baidu.com"
        #url = "http://1234.gif"
        #platform = "sina"
        platform = "iqiyi"
        t_crawl_url(url, platform)
        sys.exit(0)

    #start crawl thread
    threads = []
    for i in range(num_threads):
        t1 = CrawlThread(i)
        threads.append(t1)
        t1.start()

    time.sleep(10)

    for t in threads:
        t.join()
