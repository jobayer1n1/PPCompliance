from operator import imod
import os
import json
import csv
from pathlib import Path

chrome_api=["accessibilityFeatures", "browsingData", "commands",
            "contentSettings", "cookies", "declarativeNetRequest",
            "desktopCapture", "devtools",
            "enterprise", "fileBrowserHandler", "history",
            "identity", "instanceID", "permissions", "power", "privacy", "proxy", "runtime",
            "scripting", "storage", "system"]

tag_csv_header=['html_file','tag','id','class','text','placeholder','raw_html']

input_dom = ["get_element_operation_times", "create_element_operation_times", "other_operation_times"]

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_csv(path,data,csv_header):
    with open(path,'w') as f:
        writer = csv.writer(f)
        # write a row to the csv file
        header=['ext']+csv_header
        writer.writerow(header)
        writer.writerows(data)

def traverDOM(dir_path):
    res_list = []
    # Tell the CSV builder what our table columns will be
    global input_dom
    
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            if item[-19:] == '_dom_operation.json':
                ext_id = item[:-19]
                
                with open(os.path.join(dir_path, item), 'r') as f:
                    file_data = json.load(f)
                
                # Extract just the top-level count totals, defaulting to 0 if missing
                tmp = [
                    ext_id,
                    file_data.get("get_element_operation_times", 0),
                    file_data.get("create_element_operation_times", 0),
                    file_data.get("other_operation_times", 0)
                ]
                res_list.append(tmp)
                print('Analyzed DOM for extension:', ext_id)
                
    return res_list


def traverAPI(dir_path):
    res_list=[]
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            if item[-17:] == '_privacy_api.json':
                ext_id=item[:-17]
                tmp=[ext_id]
                with open(os.path.join(dir_path, item),'r') as f:
                    file_data=json.load(f)
                for api in chrome_api:
                    if api in file_data.keys():
                        # contains the api
                        tmp.append(file_data[api])
                    else:
                        # not contain the api
                        tmp.append(0)
                res_list.append(tmp)        
                print('analysis the ext',item)
    return res_list

def traverTags(dir_path):
    res_list=[]
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            if item[-20:] == '_userinput_tags.json':
                ext_id=item[:-20]

                with open(os.path.join(dir_path, item),'r') as f:
                    file_data=json.load(f)

                for item in file_data:
                    # traverse all html files
                    if len(item["user_input_tages"])!=0:
                        # there is user input tags
                        for tag in item["user_input_tages"]:
                            tmp=[ext_id,item['file_name'],tag['tag_name'],
                                tag['id'],tag['class'],tag['text'],tag['placeholder'],tag['raw_html']]
                            res_list.append(tmp)


                print('handle tags for ext',ext_id)
    return res_list

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    dir_path = str(script_dir.parent / 'raw_data' / 'process')
    api_csv_path = str(script_dir / 'chrome_api_conclude.csv')
    tag_csv_path = str(script_dir / 'dynamic_input_tag_conclude.csv')
    dom_ = str(script_dir / 'dom_conclude.csv')
    res_list=traverDOM(dir_path)
    save_csv(dom_,res_list,input_dom)

    res_list=traverAPI(dir_path)
    save_csv(api_csv_path,res_list,chrome_api)

    # res_list=traverTags(dir_path)
    # save_csv(tag_csv_path,res_list,tag_csv_header)