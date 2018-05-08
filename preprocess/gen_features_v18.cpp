#include <vector>
#include <algorithm>

#include <map>
#include <string>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fstream>
#include <ostream>
#include <iostream> 
#include <fstream>
#include <sstream>

#define YEAR	0
#define MONTH	1
#define DAY	2
#define HOUR	3
#define MINUTE	4
#define SECOND	5

using namespace std;

typedef map<string,float> NAMED_VALUE;
typedef map<string,vector<float>> NAMED_DATA;
typedef map<int, NAMED_DATA> DAY_DATA;
typedef map<int, DAY_DATA> MONTH_DATA ;

typedef map<int, vector<float>> DAYRANGE_DATA;
typedef map<int,DAYRANGE_DATA> DAYMONTH_DATA;

typedef map<int, NAMED_VALUE> MONTH_NAMED_VALUE;

string& trim(string &s)   
{  
    if (s.empty()) {  
	return s;  
    }  
	      
    s.erase(0,s.find_first_not_of(" \r"));  
    s.erase(0,s.find_first_not_of(" \r"));  
    s.erase(s.find_last_not_of(" \r") + 1);  
    return s;  
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while(std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2-pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if(pos1 != s.length())
		v.push_back(s.substr(pos1));
}

inline string ftoa(float n){
	char s[32];
	snprintf(s,sizeof(s),"%.4f",n);
	string ss(s);
	return ss;
}

inline string itoa(int n){
	char s[32];
	snprintf(s,sizeof(s),"%d",n);
	string ss(s);
	return ss;
}


vector<int> parse_datetime(string d){
	static const char *sep = (const char*)("- :");
	vector<int> ret;
	char *p;
	p = strtok((char*)d.c_str(), sep);
	while(p){
		ret.push_back(atoi(p));
		p = strtok(NULL,sep);
	}

	//cerr << d <<  endl;
	//cerr << ret[0] << "," << ret[1] << "," << ret[2] << ","<< ret[3] << endl;
	return ret;
}

class Counter {
public:
    int clk = 0; // clicks
    int down = 0;
    int hour_cnt[24] = {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0};
    int hour_6_cnt[6]={0,0,0,0,0,0};
    int min_4_cnt[4] = {0,0,0,0};
    string result;
};

class FeatureStat{
    string input_file;
    public:
        FeatureStat(string input_file){
            this->input_file = input_file;
        }

        virtual void add_data(map<string,string>& row){
        }
        virtual void output_data(){
        }
        void process_file(string input_file){
            //train_fp = fopen((name + "_train.libsvm").c_str(),"w");
            //test_fp = fopen((name + "_test.libsvm").c_str(),"w");

            ifstream fin(input_file.c_str());
            if (!fin) {
                cerr << "Can not open file " << input_file << endl;
                return ;
            }

            // read filed
            string line;
            fin >> line;
            // get field names
            vector<string> names;
            SplitString(line, names,",");

            vector<string> flds;
            map<string,string> row;
            int lines = 0;
            int names_size = names.size();
            while (getline(fin,line,'\n')){
                line = trim(line);
                if (line.size() <=2) continue;

                //cout << line << endl;
                flds.clear();
                row.clear();
                SplitString(line, flds,",");
                for (int i =0 ; i < names_size; i++){
                    row[names[i]] = flds[i];
                }
                add_data(row);

                lines += 1;
                if (lines >= 180903891)
                   break;
                if (lines % 100000 == 0)
                    cerr << "Processed lines: " << lines << endl;
            }
            fin.close();

            //output_data();
        }
};

class TalkingDataStat: public FeatureStat{
public:
    string ver = "18.0";

    TalkingDataStat(string input_file): FeatureStat(input_file){
    }
    //Mean counter
    Counter mean_ip_counter;
    Counter mean_app_counter;
    Counter mean_os_counter;
    Counter mean_device_counter;
    Counter mean_channel_counter;
    Counter mean_ip_app_counter;
    Counter mean_ip_app_hour_counter;
    Counter mean_ip_os_counter;
    Counter mean_ip_device_counter;
    Counter mean_ip_channel_counter;
    Counter mean_app_os_counter;
    Counter mean_app_os_hour_counter;
    Counter mean_app_device_counter;
    Counter mean_app_device_hour_counter;
    Counter mean_app_channel_counter;
    Counter mean_app_channel_hour_counter;
    Counter mean_os_device_counter;
    Counter mean_os_device_hour_counter;
    Counter mean_device_channel_counter;
    Counter mean_ip_app_os_counter;
    Counter mean_ip_app_device_counter;
    Counter mean_ip_app_channel_counter;
    Counter mean_app_os_device_counter;
    Counter mean_app_device_channel_counter;
    
    //counter map
    map<string,Counter> ip_counter;
    map<string,Counter> app_counter;
    map<string,Counter> os_counter;
    map<string,Counter> device_counter;
    map<string,Counter> channel_counter;
    map<string,Counter> ip_app_counter;
    map<string,Counter> ip_app_hour_counter;
    map<string,Counter> ip_os_counter;
    map<string,Counter> ip_device_counter;
    map<string,Counter> ip_channel_counter;
    map<string,Counter> app_os_counter;
    map<string,Counter> app_os_hour_counter;
    map<string,Counter> app_device_counter;
    map<string,Counter> app_device_hour_counter;
    map<string,Counter> app_channel_counter;
    map<string,Counter> app_channel_hour_counter;
    map<string,Counter> os_device_counter;
    map<string,Counter> os_device_hour_counter;
    map<string,Counter> device_channel_counter;
    map<string,Counter> ip_app_os_counter;
    map<string,Counter> ip_app_device_counter;
    map<string,Counter> ip_app_channel_counter;
    map<string,Counter> app_os_device_counter;
    map<string,Counter> app_device_channel_counter;

    void add_data(map<string,string>& row){
        int cnt = 1;
        int down = (row["is_attributed"] == "1")?1:0;
        vector<int> dt = parse_datetime(row["click_time"]);
        int day = dt[2];
        int hour = dt[3];
        int min = dt[4];

        string &ip = row["ip"];
        string &app = row["app"];
        string &os = row["os"];
        string &device = row["device"];
        string &channel = row["channel"];
        string h = itoa(hour);
        //cout << hour << " " << hour/4 << " " << min << " "<< min/15 << endl;
        add_number(ip_os_counter,ip + "_" + os,cnt,down,day,hour,min);

        add_number(ip_counter,ip,cnt,down,day,hour,min);
        add_number(app_counter,app,cnt,down,day,hour,min);
        add_number(os_counter,os,cnt,down,day,hour,min);
        add_number(device_counter,device,cnt,down,day,hour,min);
        add_number(channel_counter,channel,cnt,down,day,hour,min);

        add_number(ip_app_counter,ip+"_"+app,cnt,down,day,hour,min);
        add_number(ip_app_hour_counter,ip+"_"+app+"_" + h,cnt,down,day,hour,min);

        add_number(ip_os_counter,ip+"_"+os,cnt,down,day,hour,min);
        add_number(ip_device_counter,ip+"_"+device,cnt,down,day,hour,min);
        add_number(ip_channel_counter,ip+"_"+channel,cnt,down,day,hour,min);

        add_number(app_os_counter,app+"_"+os,cnt,down,day,hour,min);
        add_number(app_os_hour_counter,app+"_"+os+"_"+h,cnt,down,day,hour,min);

        add_number(app_device_counter,app+"_"+device,cnt,down,day,hour,min);
        add_number(app_device_hour_counter,app+"_"+device+"_"+h,cnt,down,day,hour,min);

        add_number(app_channel_counter,app+"_"+channel,cnt,down,day,hour,min);
        add_number(app_channel_hour_counter,app+"_"+channel+"_" +h,cnt,down,day,hour,min);

        add_number(os_device_counter,os+"_"+device,cnt,down,day,hour,min);
        add_number(os_device_hour_counter,os+"_"+device+"_"+h,cnt,down,day,hour,min);

        add_number(device_channel_counter,device+"_"+channel,cnt,down,day,hour,min);
        add_number(ip_app_os_counter,ip+"_"+app + "_" + os,cnt,down,day,hour,min);
        add_number(ip_app_device_counter,ip+"_"+app + "_" + device,cnt,down,day,hour,min);
        add_number(ip_app_channel_counter,ip+"_"+app + "_" + channel,cnt,down,day,hour,min);
        add_number(app_os_device_counter,app+"_"+os+ "_" + device,cnt,down,day,hour,min);
        add_number(app_device_channel_counter,app+"_"+device + "_" + channel,cnt,down,day,hour,min);
    }

    void add_number(map<string,Counter> & stat, string key,int cnt, int down, int day, int hour, int min) {
        map<string,Counter>::iterator it = stat.find(key);
        if (it == stat.end()){
            Counter counter;
            counter.down = down;
            counter.clk = cnt;
            counter.hour_cnt[hour] = 1;
            counter.hour_6_cnt[hour/4] =1;
            counter.min_4_cnt[min/15]=1;
            stat[key] = counter;
        }else{
            stat[key].down += down;
            stat[key].clk += cnt;
            stat[key].hour_cnt[hour] += 1;
            stat[key].hour_6_cnt[hour/4] +=1;
            stat[key].min_4_cnt[min/15] +=1;
        }
    }

    Counter calculate_mean_and_result_values_for_stat(string file_name, map<string,Counter> &stat) {
        //compute average number for stat
		
		ofstream fo((file_name+"."+ver).c_str(),ios_base::out|ios_base::trunc);
        Counter counter;
        int total_num = 0;
        map<string,Counter>::iterator it = stat.begin();
        for(;it != stat.end();it ++){
            //only for
            if(1){
                stringstream result_s ;
                result_s << "," << it->second.clk << "," << it->second.down;
                result_s << "," << float(it->second.down)/float(it->second.clk);
				float clk = it->second.clk;
                /*
                for (int i = 0 ; i < 24; i++){
                    counter.hour_cnt[i] += it->second.hour_cnt[i];
                }*/
				float entropy = 0.0;
                for (int i = 0 ; i < 6; i++){
					float c = it->second.hour_6_cnt[i];
					if (c >0 && clk >0)
							entropy = -(c/clk)*logf(c/clk);
                    //result_s << "," << it->second.hour_6_cnt[i];
                }
				result_s << "," << entropy ;
				entropy = 0.0;
                for (int i = 0 ; i < 4; i++){
					float c = it->second.min_4_cnt[i];
					if (c >0 && clk >0)
							entropy = -(c/clk)*logf(c/clk);
                    //result_s << "," << it->second.min_4_cnt[i];
                }
				result_s << "," << entropy ;
                it->second.result = result_s.str();
				//if (it->second.clk >5)
				//output all
				fo << it->first << result_s.str() << endl;
            }
            if (it->second.clk <= 10000){
                counter.clk += it->second.clk;
                counter.down += it->second.down;
                /*
                for (int i = 0 ; i < 24; i++){
                    counter.hour_cnt[i] += it->second.hour_cnt[i];
                }*/
                for (int i = 0 ; i < 6; i++){
                    counter.hour_6_cnt[i] += it->second.hour_6_cnt[i];
                }
                for (int i = 0 ; i < 4; i++){
                    counter.min_4_cnt[i] += it->second.min_4_cnt[i];
                }
                total_num += 1;
            }
        }
        stringstream result_s ;
        //counter.clk /= total_num;
        //counter.down /= total_num;
        result_s << "," << float(counter.clk)/float(total_num) << "," << float(counter.down)/float(total_num);
        result_s << "," << float(counter.down)/float(counter.clk);
        /*for (int i = 0 ; i < 24; i++){
            counter.hour_cnt[i] /= total_num;
            result_s << "," << counter.hour_cnt[i];
        }*/
		float entropy = 0.0;
		float clk = counter.clk;
        for (int i = 0 ; i < 6; i++){
			float c = counter.hour_6_cnt[i];
			if (c >0 && clk >0)
					entropy = -(c/clk)*logf(c/clk);
            //result_s << "," << counter.hour_6_cnt[i];
        }
		result_s << "," << entropy ;
		entropy = 0.0;
        for (int i = 0 ; i < 4; i++){
			float c = counter.min_4_cnt[i];
			if (c >0 && clk >0)
					entropy = -(c/clk)*logf(c/clk);
            //result_s << "," << counter.min_4_cnt[i];
        }
		result_s << "," << entropy ;

        //add mean
        counter.result = result_s.str();
        stat["mean"] = counter;
		fo << "mean" << result_s.str() << endl;
		fo.close();
        return counter;
    }

    void calculate_mean_values(){
        cout << "Calculate mean values" << endl;
        mean_ip_counter = calculate_mean_and_result_values_for_stat("grp_ip.stat",ip_counter);
        mean_app_counter = calculate_mean_and_result_values_for_stat("grp_app.stat",app_counter);
        mean_os_counter = calculate_mean_and_result_values_for_stat("grp_os.stat", os_counter);
        mean_device_counter = calculate_mean_and_result_values_for_stat("grp_device.stat",device_counter);
        mean_channel_counter = calculate_mean_and_result_values_for_stat("grp_channel.stat",channel_counter);

        mean_ip_app_counter = calculate_mean_and_result_values_for_stat("grp_ip_app.stat",ip_app_counter);
        mean_ip_app_hour_counter = calculate_mean_and_result_values_for_stat("grp_ip_app_hour.stat",ip_app_hour_counter);

        //mean_ip_os_counter = calculate_mean_and_result_values_for_stat("grp_ip_os.stat",ip_os_counter);
        //mean_ip_device_counter = calculate_mean_and_result_values_for_stat("grp_ip_device.stat",ip_device_counter);
        //mean_ip_channel_counter = calculate_mean_and_result_values_for_stat("grp_ip_channel.stat",ip_channel_counter);
        mean_app_os_counter = calculate_mean_and_result_values_for_stat("grp_app_os.stat",app_os_counter);
        mean_app_os_hour_counter = calculate_mean_and_result_values_for_stat("grp_app_os_hour.stat",app_os_hour_counter);

        mean_app_device_counter = calculate_mean_and_result_values_for_stat("grp_app_device.stat",app_device_counter);
        mean_app_device_hour_counter = calculate_mean_and_result_values_for_stat("grp_app_device_hour.stat",app_device_hour_counter);

        mean_app_channel_counter = calculate_mean_and_result_values_for_stat("grp_app_channel.stat",app_channel_counter);
        mean_app_channel_hour_counter = calculate_mean_and_result_values_for_stat("grp_app_channel_hour.stat",app_channel_hour_counter);

        mean_os_device_counter = calculate_mean_and_result_values_for_stat("grp_os_device.stat",os_device_counter);
        mean_os_device_hour_counter = calculate_mean_and_result_values_for_stat("grp_os_device_hour.stat",os_device_hour_counter);
        //mean_device_channel_counter = calculate_mean_and_result_values_for_stat("grp_device_channel.stat",device_channel_counter);
        //mean_ip_app_os_counter = calculate_mean_and_result_values_for_stat("grp_ip_app_os.stat",ip_app_os_counter);
        //mean_ip_app_device_counter = calculate_mean_and_result_values_for_stat("grp_ip_app_device.stat",ip_app_device_counter);
        //mean_ip_app_channel_counter = calculate_mean_and_result_values_for_stat("grp_ip_app_channel.stat",ip_app_channel_counter);
        //mean_app_os_device_counter = calculate_mean_and_result_values_for_stat("grp_app_os_device.stat",app_os_device_counter);
        //mean_app_device_channel_counter = calculate_mean_and_result_values_for_stat("grp_app_device_channel.stat",app_device_channel_counter);
    }

    void check_counters(){
        map<string,Counter>& stat = ip_device_counter;
        map<string,Counter>::iterator it = stat.begin();
        for(;it != stat.end();it ++){
            string ret = it->first;
            //only for
            if (it->second.clk < 5000 && it->second.clk > 5){
                ret += " " + itoa(it->second.clk) ;
                ret += " " + itoa(it->second.down);
                for (int i = 0 ; i < 24; i++){
                    ret += " " + itoa(it->second.hour_cnt[i]);
                }
                for (int i = 0 ; i < 6; i++){
                    ret += " " + itoa(it->second.hour_6_cnt[i]);
                }
                for (int i = 0 ; i < 4; i++){
                    ret += " " + itoa(it->second.min_4_cnt[i]);
                }
            }
            cout << "key=" << ret << endl;
        }
    }

    string last_key;
    Counter last_counter;
    void add_feature_by_key(
        string key,
        map<string,Counter> counter_map,
        Counter & mean,
        map<string,string>&row,
        ofstream & fo)
    {
        map<string,Counter>::iterator it = counter_map.find(key);
        if (key == last_key){
        }else{
            last_key = key;
            if (it == counter_map.end() || it->second.clk <5){
                last_counter = mean;
            }else{
                last_counter = it->second;
            }
        }
        Counter * p_counter = &last_counter;
        fo << p_counter->result ;
    }

    void gen_features_for_row(map<string,string> & row, ofstream & fo,
		    string &ip, string & app, string &os,
		    string &device, string & channel, string &hour) {
        string key ;

        key = ip;
        add_feature_by_key(key, ip_counter, mean_ip_counter, row,fo);
        key = app;
        add_feature_by_key(key, app_counter,mean_app_counter, row,fo);
        key = os;
        add_feature_by_key(key, os_counter,mean_os_counter, row,fo);
        key = device;
        add_feature_by_key(key, device_counter,mean_device_counter, row,fo);
        key = channel;
        add_feature_by_key(key, channel_counter,mean_channel_counter, row,fo);

        key = ip+"_"+app;
        add_feature_by_key(key, ip_app_counter,mean_ip_app_counter, row,fo);

        key = ip+"_"+app + "_" + hour;
        add_feature_by_key(key, ip_app_hour_counter,mean_ip_app_hour_counter, row,fo);
        //key = ip+"_"+os;
        //add_feature_by_key(key, ip_os_counter,mean_ip_os_counter,row,fo);
        //key = ip+"_"+device;
        //add_feature_by_key(key, ip_device_counter,mean_ip_device_counter,row,fo);
        //key = ip+"_"+channel;
        //add_feature_by_key(key, ip_channel_counter,mean_ip_channel_counter,row,fo);
        key = app+"_"+os;
        add_feature_by_key(key, app_os_counter,mean_app_os_counter,row,fo);

        key = app+"_"+os + "_" + hour;
        add_feature_by_key(key, app_os_hour_counter,mean_app_os_hour_counter,row,fo);

        key = app+"_"+device;
        add_feature_by_key(key, app_device_counter,mean_app_device_counter,row,fo);

        key = app+"_"+device + "_" + hour;
        add_feature_by_key(key, app_device_hour_counter,mean_app_device_hour_counter,row,fo);

        key = app+"_"+channel;
        add_feature_by_key(key, app_channel_counter,mean_app_channel_counter,row,fo);

        key = app+"_"+channel + "_" + hour;
        add_feature_by_key(key, app_channel_hour_counter,mean_app_channel_hour_counter,row,fo);

        key = os+"_"+device;
        add_feature_by_key(key, os_device_counter,mean_os_device_counter,row,fo);

        key = os+"_"+device + "_" + hour;
        add_feature_by_key(key, os_device_hour_counter,mean_os_device_hour_counter,row,fo);
        //key = device+"_"+channel;
        //add_feature_by_key(key, device_channel_counter,mean_device_channel_counter,row,fo);
        //key = ip+"_"+app + "_" + os;
        //add_feature_by_key(key, ip_app_os_counter,mean_ip_app_os_counter,row,fo);
        //key = ip+"_"+app + "_" + device;
        //add_feature_by_key(key, ip_app_device_counter,mean_ip_app_device_counter,row,fo);
        //key = ip+"_"+app + "_" + device + "_" + hour;
        //add_feature_by_key(key, ip_app_device_hour_counter,mean_ip_app_device_hour_counter,row,fo);
        //key = ip"+"_"+app + "_" + channel;
        //add_feature_by_key(key, ip_app_channel_counter,mean_ip_app_channel_counter,row,fo);
        //key = app"+"_"+os+ "_" + device;
        //add_feature_by_key(key, app_os_device_counter,mean_app_os_device_counter,row,fo);
        //key = app+"_"+device + "_" + channel;
        //add_feature_by_key(key, app_device_channel_counter,mean_app_device_channel_counter,row,fo);
    }

    void do_gen_features_from_file(string fn,int start,int is_train) {
            ifstream fin(fn.c_str());
            if (!fin) {
                cerr << "Can not open file " << fn << endl;
                return ;
            }

            ofstream fo((fn+".gen."+ver).c_str(),ios_base::out|ios_base::trunc);
            // read filed
            string line;
            fin >> line;
            // get field names
            vector<string> names;
            SplitString(line, names,",");

            vector<string> flds;
            map<string,string> row;
            int lines = 0;

            vector<float> ratio_ret;
            vector<int> ret;
            int names_size = names.size();
            while (getline(fin,line,'\n')){
                lines += 1;
                if (lines <start){
                    continue;
                }

                line = trim(line);
                if (line.size() <=2) continue;

                //cout << line << endl;
                flds.clear();
                row.clear();
                SplitString(line, flds,",");
                for (int i =0 ; i < names_size; i++){
                    row[names[i]] = flds[i];
                }
                string &click_time = row["click_time"];
                string &ip = row["ip"];
                string &app = row["app"];
                string &os = row["os"];
                string &device = row["device"];
                string &channel = row["channel"];
                int down = row["is_attributed"] == "1" ? 1:0;

                vector<int> dt = parse_datetime(click_time);
                string hour = itoa(dt[3]);
                if(is_train){
                    //int h= dt[3];
                    //if(!(h == 4||h==5 ||h==9 ||h==10||h==13||h==14))
                    //    continue;
                    fo << down <<"," << ip<< ","<< app << "," << os
                        << "," << device << "," << channel
                        << "," << dt[2] << "," << dt[3] << "," << int(dt[3]/4) << "," << int(dt[4]/15);
                    gen_features_for_row(row, fo, ip, app, os, device, channel,hour);
                }else{
                    fo << row["click_id"] <<"," << ip<< ","<< app << "," << os
                        << "," << device << "," << channel
                        << "," << dt[2] << "," << dt[3] << "," << int(dt[3]/4) << "," << int(dt[4]/15);
                    gen_features_for_row(row, fo, ip, app, os, device, channel,hour);
                }
                fo << endl;

                //if (lines==48){
                //    cerr << "line:" << lines << ", line=" << line << endl;
                //}
                //if (lines > 10000)
                //    break;
                if (lines % 10000 == 0)
                    cerr << "Processed lines: " << lines << endl;
            }
            fin.close();
            fo.close();
    }

    void do_gen_features(){
        cout << "Gen features for train" << endl;
        //use only a portion of the data
        do_gen_features_from_file("data/train.csv",131886954,1);
        cout << "Gen features for test " << endl;
        do_gen_features_from_file("data/test.csv",0,1);
    }

    void output_data(){
        calculate_mean_values();

        //check counter
        //check_counters();
        //gen features for train data and test data
        //do_gen_features();
    }
};

int main(int argc, char ** argv){
	cerr << "Process Begins " << endl;
	string fn = "newdata/train_big.csv";
	TalkingDataStat stat(fn);
	stat.process_file(fn);
	fn = "newdata/dev1.csv";
	stat.process_file(fn);
	fn = "newdata/dev2.csv";
	stat.process_file(fn);
	fn = "newdata/test.csv";
	stat.process_file(fn);
	stat.output_data();
}
