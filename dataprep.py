import numpy as np
from xml.sax import parse, handler
import os
import random, shutil
import pandas as pd
import math

class DetectorParse(handler.ContentHandler):
    """reading in the dataset from xml file
    """
    def __init__(self):
        self.lanes = {}
        self.sample = {}

    def startElement(self, name, attrs):
        if name == 'timestep':
            self.probe_id = attrs['id']


        if name == 'vehicle':

          pos = float(attrs['pos'])

          pos_i = str(int(pos//1000))
          lane_id = attrs['lane']+'%'+pos_i

          if lane_id not in self.lanes:
              self.lanes[lane_id] = {}
              self.lanes[lane_id]['Speed'] = []
              self.lanes[lane_id]['ID'] = []
              self.lanes[lane_id]['AverageSpeed'] = []
              self.sample[lane_id] = {}
              self.sample[lane_id]['Original'] = []
              self.sample[lane_id]['Attacker'] = []
          self.lanes[lane_id]['Speed'].append(float(attrs['speed']))
          self.lanes[lane_id]['ID'].append(str(attrs['id']))


    def endElement(self, name):
        if name == 'timestep':
            if self.probe_id == 'probe2':
              for key, value in self.lanes.items():
                  n = len(value['Speed'])
                  ave_speed = []

                  if n > 0:
                      rate = 0.3  # sample of benign users
                      picknumber = math.ceil(n * rate)
                      sample = random.sample(list(zip(value['Speed'], value['ID'])), picknumber)
                      self.sample[key]['Original'].append(sample)
                      speed, id = zip(*sample)
                      std = np.std(speed)
                      ave_speed = np.mean(speed)

                  if n > 1:
                      rate_att = 0.3  # sample of attackers
                      picknumber_att = math.ceil(n * rate_att)
                      remain_index = tuple(set(range(len(list(zip(value['Speed'], value['ID']))))) - set(sample))
                      att_index = random.sample(remain_index, picknumber_att)
                      sample_att = [list(zip(value['Speed'], value['ID']))[i] for i in att_index]
                      self.sample[key]['Original'].append(sample_att)

                      speed_att, id_att = zip(*sample_att)
                      speed_att_new = np.random.normal(5, std, len(speed_att))
                      ave_speed_tmp = []
                      ave_speed_tmp.append(speed)
                      ave_speed_tmp.append(speed_att_new)
                      ave_speed = np.mean(ave_speed_tmp)
                      self.sample[key]['Attacker'].append(list(zip(speed_att_new, id_att)))

                  self.lanes[key]['AverageSpeed'].append(ave_speed)
                  self.lanes[key]['Speed'] = []


class DetectorParse_attacker(handler.ContentHandler):
    """reading in the dataset from xml file
    """

    def __init__(self):
        self.lanes = {}
        self.sample = {}
        self.lane_list = ['56250003#0_0%1', '237111029#1.254_0%0', '237111029#1.254_0%1', '237111029#1.254_1%0',
                          '237111029#1.254_1%1', '238459506.6_0%1', '62830645#2.0.0_0%0', '62830645#2.0.0_0%1',
                          '62830645#2.0.0_1%0', '62830645#2.0.0_1%1']

    def startElement(self, name, attrs):
        if name == 'timestep':
            self.probe_id = attrs['id']
            self.time = float(attrs['time'])

        if name == 'vehicle':
            pos = float(attrs['pos'])
            pos_i = str(int(pos // 1000))
            lane_id = attrs['lane'] + '%' + pos_i
            if lane_id in self.lane_list and lane_id not in self.lanes:
                self.lanes[lane_id] = {}
                self.lanes[lane_id]['Speed'] = []
                self.lanes[lane_id]['ID'] = []
                # self.lanes[lane_id]['AverageSpeed'] = []
                self.sample[lane_id] = {}
                self.sample[lane_id]['Original'] = []
                self.sample[lane_id]['Attacker'] = []
            if (lane_id in ['56250003#0_0%1', '237111029#1.254_0%0', '237111029#1.254_0%1', '237111029#1.254_1%0','237111029#1.254_1%1'] and self.time >= 10800.00 and self.time <= 13200.00) or \
                    (lane_id in ['238459506.6_0%1', '62830645#2.0.0_0%0', '62830645#2.0.0_0%1', '62830645#2.0.0_1%0','62830645#2.0.0_1%1'] and self.time >= 20400.00 and self.time <= 21600.00):
                self.lanes[lane_id]['Speed'].append(float(attrs['speed']))
                self.lanes[lane_id]['ID'].append(str(attrs['id']))

    def endElement(self, name):
        if name == 'timestep':
            if self.probe_id == 'probe2':
                for key, value in self.lanes.items():
                    if (key in ['56250003#0_0%1', '237111029#1.254_0%0', '237111029#1.254_0%1', '237111029#1.254_1%0','237111029#1.254_1%1'] and self.time >= 10800.00 and self.time <= 13200.00) or \
                            (key in ['238459506.6_0%1', '62830645#2.0.0_0%0', '62830645#2.0.0_0%1', '62830645#2.0.0_1%0','62830645#2.0.0_1%1'] and self.time >= 20400.00 and self.time <= 21600.00):
                        # if (key in ['237111029#1.254_0%0','237111029#1.254_0%1'] and self.time >= 10800.00 and self.time <= 13200.00):

                        # 15:00-15:40 #17:40-18:00
                        n = len(value['Speed'])
                        if n == 1:
                            # rate = 0.3  # sample of benign users
                            # picknumber = math.ceil(n * rate)
                            # sample = random.sample(list(zip(value['Speed'], value['ID'])), picknumber)
                            self.sample[key]['Original'].append(list(zip(value['Speed'], value['ID'])))
                            self.sample[key]['Attacker'].append('None')

                            # ave_speed = np.mean(speed)

                        elif n > 1:
                            rate = 0.3  # sample of benign users
                            picknumber = math.ceil(n * rate)
                            sample = random.sample(list(zip(value['Speed'], value['ID'])), picknumber)
                            # self.sample[key]['Original'].append(sample)
                            speed, id = zip(*sample)
                            std = np.std(speed)

                            rate_att = 0.3  # sample of attackers
                            picknumber_att = math.ceil(n * rate_att)
                            value['ID'] += 'att'
                            remain_index = tuple(set(range(len(list(zip(value['Speed'], value['ID']))))) - set(sample))
                            att_index = random.sample(remain_index, picknumber_att)
                            sample_att = [list(zip(value['Speed'], value['ID']))[i] for i in att_index]
                            # self.sample[key]['Original'].append(sample_att)

                            speed_att, id_att = zip(*sample_att)
                            speed_att_new = np.random.normal(5, 0.5, len(speed_att))
                            '''
                            ave_speed_tmp = []
                            ave_speed_tmp.append(speed)
                            ave_speed_tmp.append(speed_att_new)
                            ave_speed = np.mean(ave_speed_tmp)
                            '''
                            sample += list(zip(speed_att_new, id_att))
                            self.sample[key]['Original'].append(sample)
                            self.sample[key]['Attacker'].append(id_att)

                        # self.lanes[key]['AverageSpeed'].append(ave_speed)
                        self.lanes[key]['Speed'] = []
                        self.lanes[key]['ID'] = []

    def endDocument(self):
        # print(self.lanes)
        print(self.sample)


def connect2oneday(day, load_dict_1, load_dict_2, lane_list):

    final = []
    dicic = []
    a = []
    for i in range(len(lane_list)):
      for key in load_dict_1:
        #if key in load_dict_2:
          if  key == lane_list[i]:
            # print('key',key)
            a = load_dict_1[key]['AverageSpeed'][-720:]
            b = load_dict_2[key]['AverageSpeed'][:2160]
            a.extend(b)

            dicic.append(str(key))
            dicic.append(a)

            final.append(dicic)
            dicic = []
    print(len(final)) # (630, 1+720)

    # replace Nan
    df = final
    a = []
    n = []
    df_new = df
    for i in range(len(df)):
        a = pd.DataFrame(df[i][1])
        n = a.ffill()
        n = n.bfill().values.tolist()
        df_new[i][1] = list(np.array(n).ravel())
        p = np.isnan(n)
        if p is True:
            print(i)
        a = []
        n = []
    np.save('rawdata/npy/'+day+'_8h.npy', df_new)
    return(df_new)


def main():
    input_path = 'rawdata/'
    files = os.listdir(input_path)
    files.sort()
    # for file in files:
    for file in ['d16_0-12.xml', 'd16_12-24.xml', 'd17_0-12.xml', 'd17_12-24.xml', 'd20_0-12.xml', 'd20_12-24.xml']:
        if os.path.splitext(file)[1] == '.xml':
            save_dir = 'rawdata/npy/' + os.path.splitext(file)[0]
            detector_output = DetectorParse()
            #detector_output = DetectorParse_attacker()
            parse(input_path + file, detector_output)
            np.save(save_dir+ '.npy', detector_output.lanes)

    data = {}
    lane_list = np.load('net/lane_list.npy')
    for day in ['d16', 'd17', 'd20']:
        load_dict_1 = np.load('rawdata/npy/'+day+'_0-12.npy', allow_pickle=True).item()
        load_dict_2 = np.load('rawdata/npy/'+day+'_12-24.npy', allow_pickle=True).item()
        d_day = connect2oneday(day, load_dict_1, load_dict_2, lane_list)
        data[day] = d_day

    final = data['d20']
    np.save('rawdata/npy/8h_d20_fill.npy', final)


if __name__ == "__main__":
    main()