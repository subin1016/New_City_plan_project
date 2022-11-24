#!/usr/bin/env python
# coding: utf-8

# ### 택배송장정보 데이터 프레임 불러옴

# In[1]:


from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# In[2]:


#데이터 프레임의 모든 열 보여주기 설정
#pd.set_option('.max_columns', None)
#pd.set_option('display.max_rows', None)
#다시 요약해서 보여주는 설정
#pd.options.display.max_rows=60


# In[2]:


nums = ['01','02','03','04','05','06','07','08','09','10','11','12']
d_df = pd.DataFrame()
for num in tqdm(nums):
    d_path = 'import_data/TB_CJO_INVOICE/TB_CJO_INVOICE_2020%s.csv' %num
    d_data = pd.read_csv(d_path)
    d_df = pd.concat([d_df, d_data])


# ### 카드 소비 데이터 프레임 불러옴

# In[4]:


nums = [1,2,3,4,5,6,7,8,9]
c_spnd_df = pd.DataFrame()
for num in tqdm(nums):
    c_spnd_path = 'import_data/TB_KCB_CARD_SPND_EMD/TB_KCB_CARD_SPND_EMD_20200%d.csv' %num
    c_spnd_data = pd.read_csv(c_spnd_path)
    c_spnd_df = pd.concat([c_spnd_df, c_spnd_data])


# In[5]:


c_spnd_path = 'import_data/TB_KCB_CARD_SPND_EMD/TB_KCB_CARD_SPND_EMD_202010.csv'
c_spnd_data = pd.read_csv(c_spnd_path)
c_spnd_df = pd.concat([c_spnd_df, c_spnd_data])


# In[6]:


nums = [11,12]
for num in tqdm(nums):
    c_spnd_path = 'import_data/TB_KCB_CARD_SPND_EMD/TB_KCB_CARD_SPND_EMD_2019%d.csv' %num
    c_spnd_data = pd.read_csv(c_spnd_path)
    c_spnd_df = pd.concat([c_spnd_df, c_spnd_data])


# ### 소득 데이터 프레임 불러옴

# In[7]:


nums = [1,2,3,4,5,6,7,8,9]
i_emd_df = pd.DataFrame()
for num in tqdm(nums):
    i_emd_path = 'import_data/TB_KCB_INCOME_EMD/TB_KCB_INCOME_EMD_20200%d.csv' %num
    i_emd_data = pd.read_csv(i_emd_path)
    i_emd_df = pd.concat([i_emd_df, i_emd_data])


# In[8]:


i_emd_path = 'import_data/TB_KCB_INCOME_EMD/TB_KCB_INCOME_EMD_202010.csv'
i_emd_data = pd.read_csv(i_emd_path)
i_emd_df = pd.concat([i_emd_df, i_emd_data])


# In[9]:


nums = [11,12]
for num in tqdm(nums):
    i_emd_path = 'import_data/TB_KCB_INCOME_EMD/TB_KCB_INCOME_EMD_2019%d.csv' %num
    i_emd_data = pd.read_csv(i_emd_path)
    i_emd_df = pd.concat([i_emd_df, i_emd_data])


# In[10]:


i_emd_df_cp = i_emd_df.copy()


# ### 자택직장정보 데이터 프레임 불러옴

# In[11]:


od_commu = pd.DataFrame()

for i in (3,6,9):
    
    od_commu_path = 'import_data/TB_KCB_OD_COMMU/TB_KCB_OD_COMMU_20200%d.csv' %i
    od_commu_data = pd.read_csv(od_commu_path)
    od_commu_cp_1 = od_commu_data.copy()
    od_commu_cp_1.기준년월= '20200%d' %(i-1)
    od_commu_cp_2 = od_commu_data.copy()
    od_commu_cp_2.기준년월= '20200%d' %(i-2)
    od_commu = pd.concat([od_commu, od_commu_cp_2])
    od_commu = pd.concat([od_commu, od_commu_cp_1])
    od_commu = pd.concat([od_commu, od_commu_data])

od_commu_path = 'import_data/TB_KCB_OD_COMMU/TB_KCB_OD_COMMU_201912.csv'

od_commu_data = pd.read_csv(od_commu_path)

for i in (10,11,12):
    od_commu_cp = od_commu_data.copy()
    od_commu_cp.기준년월= '2020%d' %i
    od_commu = pd.concat([od_commu, od_commu_cp])
    
od_commu = od_commu[od_commu.법정동자택주소.astype('str').str[0:2] == '11']

od_commu.reset_index(inplace= True)

od_commu = od_commu[['기준년월','법정동자택주소','집계인구수','월환산평균소득금액','월환산중위소득금액','월평균카드소비금액','경제활동인구수']]


# ## 유동인구 데이터 불러옴

# In[12]:


nums = ['01','02','03','04','05','06','07','08','09','10','11','12']
ktc_df = pd.DataFrame()
for num in tqdm(nums):
    ktc_path = 'import_data/TB_KTC_FLOW_POPULATION/TB_KTC_FLOW_POPULATION_2019%s.csv' %num
    ktc_data = pd.read_csv(ktc_path)
    ktc_df = pd.concat([ktc_df, ktc_data])


# ## 송장정보 데이터 전처리

# In[13]:


d_df_cp = d_df.copy()


# In[14]:


d_df_cp = d_df_cp[['기준일자','시도명','시군구명','시군구코드','읍면동명','법정동코드','주문건수']]
d_seoul = d_df_cp[d_df_cp['시도명'].str.startswith("서울")]
d_seoul.drop(['시도명'],axis=1)
d_seoul.reset_index(inplace= True)
d_seoul.drop(['index','기준일자'],axis=1)


# In[15]:


d_order = d_seoul.groupby(by=['법정동코드','읍면동명'])['주문건수'].sum()
d_order = pd.DataFrame(d_order)
d_order.reset_index(inplace=True)


# ## 카드 소비데이터 전처리

# In[16]:


c_spnd_df_cp = c_spnd_df.copy()

spnd_emd = c_spnd_df_cp[c_spnd_df_cp.읍면동코드.astype('str').str[0:2] == '11']

spnd_emd.reset_index(inplace= True)

spnd_emd = spnd_emd[['기준년월','읍면동코드','읍면동명','성별구분코드','십단위연령구분코드','평균3개월카드이용금액']]

spnd_emd_mean = spnd_emd.groupby(by=['읍면동코드','읍면동명','성별구분코드','십단위연령구분코드'])['평균3개월카드이용금액'].mean()

spnd_emd_mean = pd.DataFrame(spnd_emd_mean)

spnd_emd_mean.reset_index(inplace=True)


# In[17]:


for i in (1,2):
    for j in (1,2,3,4,5,6):
        s = str(i)+'/'+str(j)+'/평균3개월카드이용금액'
        spnd_emd_mean[s] = np.where((spnd_emd_mean['성별구분코드'] == i) & (spnd_emd_mean['십단위연령구분코드'] == j), spnd_emd_mean['평균3개월카드이용금액'], 0)


# In[19]:


spnd_emd_group = spnd_emd_mean.groupby(by=['읍면동코드','읍면동명'])['1/1/평균3개월카드이용금액','1/2/평균3개월카드이용금액','1/3/평균3개월카드이용금액','1/4/평균3개월카드이용금액','1/5/평균3개월카드이용금액','1/6/평균3개월카드이용금액',
                                                               '2/1/평균3개월카드이용금액','2/2/평균3개월카드이용금액','2/3/평균3개월카드이용금액','2/4/평균3개월카드이용금액','2/5/평균3개월카드이용금액','2/6/평균3개월카드이용금액'].max()


# ## 소득 데이터 전처리

# In[20]:


i_emd_cp = i_emd_df.copy()


# In[22]:


income_emd = i_emd_df_cp[i_emd_df_cp.읍면동코드.astype('str').str[0:2] == '11']
income_emd.reset_index(inplace= True)
income_emd = income_emd[['기준년월','읍면동코드','읍면동명','성별구분코드','십단위연령구분코드','평균연소득금액','상위소득자평균연소득금액','중위연소득금액']]
income_emd_mean = income_emd.groupby(by=['읍면동코드','읍면동명','성별구분코드','십단위연령구분코드'])['평균연소득금액','상위소득자평균연소득금액','중위연소득금액'].mean()
income_emd_mean = pd.DataFrame(income_emd_mean)
income_emd_mean.reset_index(inplace=True)


# In[23]:


for i in (1,2):
    for j in (1,2,3,4,5,6):
        s = str(i)+'/'+str(j)+'/평균연소득금액'
        income_emd_mean[s] = np.where((income_emd_mean['성별구분코드'] == i) & (income_emd_mean['십단위연령구분코드'] == j), income_emd_mean['평균연소득금액'], 0)
for i in (1,2):
    for j in (1,2,3,4,5,6):
        s = str(i)+'/'+str(j)+'/상위소득자평균연소득금액'
        income_emd_mean[s] = np.where((income_emd_mean['성별구분코드'] == i) & (income_emd_mean['십단위연령구분코드'] == j), income_emd_mean['상위소득자평균연소득금액'], 0)
for i in (1,2):
    for j in (1,2,3,4,5,6):
        s = str(i)+'/'+str(j)+'/중위연소득금액'
        income_emd_mean[s] = np.where((income_emd_mean['성별구분코드'] == i) & (income_emd_mean['십단위연령구분코드'] == j), income_emd_mean['중위연소득금액'], 0)
income_emd_group = income_emd_mean.groupby(by=['읍면동코드','읍면동명'])['1/1/평균연소득금액','1/2/평균연소득금액','1/3/평균연소득금액','1/4/평균연소득금액','1/5/평균연소득금액','1/6/평균연소득금액',
                                                       '2/1/평균연소득금액','2/2/평균연소득금액','2/3/평균연소득금액','2/4/평균연소득금액','2/5/평균연소득금액','2/6/평균연소득금액',
                                                       '1/1/상위소득자평균연소득금액','1/2/상위소득자평균연소득금액','1/3/상위소득자평균연소득금액','1/4/상위소득자평균연소득금액',
                                                        '1/5/상위소득자평균연소득금액','1/6/상위소득자평균연소득금액',
                                                       '2/1/상위소득자평균연소득금액','2/2/상위소득자평균연소득금액','2/3/상위소득자평균연소득금액','2/4/상위소득자평균연소득금액',
                                                        '2/5/상위소득자평균연소득금액','2/6/상위소득자평균연소득금액',
                                                       '1/1/중위연소득금액','1/2/중위연소득금액','1/3/중위연소득금액','1/4/중위연소득금액','1/5/중위연소득금액','1/6/중위연소득금액',
                                                       '2/1/중위연소득금액','2/2/중위연소득금액','2/3/중위연소득금액','2/4/중위연소득금액','2/5/중위연소득금액','2/6/중위연소득금액'].max()


# ## 자택직장 정보 전처리

# In[24]:


# 기준년월을 모두 합쳐 법정동 별로 평균 냄
commu_sum = od_commu.groupby(by=['기준년월','법정동자택주소'])['집계인구수','경제활동인구수'].sum()
commu_sum = pd.DataFrame(commu_sum)
commu_sum.reset_index(inplace=True)
commu_merge = commu_sum.groupby(by=['법정동자택주소'])['집계인구수','경제활동인구수'].mean()
commu_merge.reset_index(inplace=True)
commu_merge['직장인구수비율'] = commu_merge['경제활동인구수'] / commu_merge['집계인구수']
commu_merge['자택인구수비율'] = (commu_merge['집계인구수'] - commu_merge['경제활동인구수']) / commu_merge['집계인구수']
commu_merge = commu_merge[['법정동자택주소','집계인구수','경제활동인구수','직장인구수비율','자택인구수비율']]


# ## 인구 데이터 전처리

# In[26]:


# 인구수 데이터 불러오고 복사
p_df = pd.read_csv('upload/population.csv',encoding='cp949')
p_df_cp = p_df.copy()


# In[27]:


p_df_cp = p_df_cp[['법정동코드', '읍면동명','계','남자','여자','10대 남자','20대 남자','30대 남자','40대 남자','50대 남자','60대 남자','10대 여자','20대 여자','30대 여자','40대 여자','50대 여자','60대 여자']]

p_df_cp['계(고령제외)'] = p_df_cp['10대 남자'] + p_df_cp['20대 남자'] + p_df_cp['30대 남자'] + p_df_cp['40대 남자'] + p_df_cp['50대 남자'] + p_df_cp['60대 남자'] +p_df_cp['10대 여자'] + p_df_cp['20대 여자'] + p_df_cp['30대 여자'] + p_df_cp['40대 여자'] + p_df_cp['50대 여자'] + p_df_cp['60대 여자']
# p_df_cp['남자 총 인구 수(고령 제외)'] = p_df_cp['10대 남자'] + p_df_cp['20대 남자'] + p_df_cp['30대 남자'] + p_df_cp['40대 남자'] + p_df_cp['50대 남자'] + p_df_cp['60대 남자']
# p_df_cp['여자 총 인구 수(고령 제외)'] = p_df_cp['10대 여자'] + p_df_cp['20대 여자'] + p_df_cp['30대 여자'] + p_df_cp['40대 여자']|+ p_df_cp['50대 여자'] + p_df_cp['60대 여자']

p_df_cp['10_m_r'] = p_df_cp['10대 남자']/p_df_cp['계(고령제외)']
p_df_cp['20_m_r'] = p_df_cp['20대 남자']/p_df_cp['계(고령제외)']
p_df_cp['30_m_r'] = p_df_cp['30대 남자']/p_df_cp['계(고령제외)']
p_df_cp['40_m_r'] = p_df_cp['40대 남자']/p_df_cp['계(고령제외)']
p_df_cp['50_m_r'] = p_df_cp['50대 남자']/p_df_cp['계(고령제외)']
p_df_cp['60_m_r'] = p_df_cp['60대 남자']/p_df_cp['계(고령제외)']

p_df_cp['10_f_r'] = p_df_cp['10대 여자']/p_df_cp['계(고령제외)']
p_df_cp['20_f_r'] = p_df_cp['20대 여자']/p_df_cp['계(고령제외)']
p_df_cp['30_f_r'] = p_df_cp['30대 여자']/p_df_cp['계(고령제외)']
p_df_cp['40_f_r'] = p_df_cp['40대 여자']/p_df_cp['계(고령제외)']
p_df_cp['50_f_r'] = p_df_cp['50대 여자']/p_df_cp['계(고령제외)']
p_df_cp['60_f_r'] = p_df_cp['60대 여자']/p_df_cp['계(고령제외)']

p_df_cp = p_df_cp[['법정동코드','읍면동명','10_m_r','20_m_r','30_m_r','40_m_r','50_m_r','60_m_r','10_f_r','20_f_r','30_f_r','40_f_r','50_f_r','60_f_r']]

# 인구비율..특징 추가...?
# p_df_cp['고령 비율'] = (p_df_cp['계'] - p_df_cp['계(고령제외)']) / p_df_cp['계']


# ### 유동인구 데이터 전처리

# In[28]:


ktc_df_cp = ktc_df.copy()


# In[29]:


ktc_df_cp = pd.DataFrame(ktc_df_cp)
ktc_df_cp = ktc_df_cp[['광역시도명','시군구명','행정동코드','행정동명','시간구분코드','유동인구총합계']]
ktc_df_cp = ktc_df_cp[ktc_df_cp['광역시도명'].str.startswith("서울")]
ktc_df_cp = ktc_df_cp.drop(['광역시도명'],axis=1)
ktc_df_cp.reset_index(inplace= True)


# In[30]:


# 시간대별 유동인구만 따로 잘라냄
for i in range(0,24):
    s = str(i)+'시 유동인구'
    ktc_df_cp[s] = np.where(ktc_df_cp['시간구분코드'] == i, ktc_df_cp['유동인구총합계'], 0)
ktc_time_flow = ktc_df_cp.groupby(by=['행정동코드','행정동명'])['0시 유동인구','1시 유동인구','2시 유동인구','3시 유동인구','4시 유동인구','5시 유동인구',
                                                  '6시 유동인구','7시 유동인구','8시 유동인구','9시 유동인구','10시 유동인구','11시 유동인구','12시 유동인구',
                                                  '13시 유동인구','14시 유동인구','15시 유동인구','16시 유동인구','17시 유동인구','18시 유동인구','19시 유동인구',
                                                   '20시 유동인구','21시 유동인구','22시 유동인구','23시 유동인구'].max()
#ktc_time_flow = pd.DataFrame(ktc_time_flow)


# In[31]:


df1=ktc_time_flow.sum(axis=1)
#ktc_time_flow = pd.DataFrame(ktc_time_flow)
ktc_new_sum=pd.DataFrame(df1)
ktc_new_sum.reset_index(inplace=True)
ktc_new_sum=ktc_new_sum.rename(columns={0:'유동인구총합계'})


# In[32]:


flow_population = pd.merge(left=ktc_new_sum, right = ktc_time_flow, how = "inner", on = ['행정동코드','행정동명'])


# In[33]:


# 행정동을 법정동으로 변경
adm_law = pd.read_csv("upload/code.csv")
flow_population = pd.merge(left=adm_law, right = flow_population, how = "inner", left_on = 'adm_code', right_on='행정동코드')
flow_population = pd.DataFrame(flow_population)
code_su = flow_population['행정동코드'].value_counts()
code_su = pd.DataFrame(code_su)
code_su.reset_index(inplace=False)


# In[34]:


col_names = flow_population.columns[4:]


# In[35]:


for j in (col_names):
    a=[]
    for i in range(len(flow_population)):
        a.append(flow_population[j][i] / code_su['행정동코드'][flow_population['행정동코드'][i]])
    flow_population[j] = a


# In[36]:


filtered_f_pop = flow_population.groupby(by=['law_code'])['유동인구총합계','0시 유동인구','1시 유동인구','2시 유동인구','3시 유동인구','4시 유동인구','5시 유동인구',
                                                  '6시 유동인구','7시 유동인구','8시 유동인구','9시 유동인구','10시 유동인구','11시 유동인구','12시 유동인구',
                                                  '13시 유동인구','14시 유동인구','15시 유동인구','16시 유동인구','17시 유동인구','18시 유동인구','19시 유동인구',
                                                   '20시 유동인구','21시 유동인구','22시 유동인구','23시 유동인구'].sum()
filtered_f_pop.reset_index(inplace=True)
for i in range(0,24):
    filtered_f_pop['%s시 유동인구' %i] = filtered_f_pop['%s시 유동인구'%i]/filtered_f_pop['유동인구총합계']


# ### 인구수 관련 데이터와 주문건수 MERGE -> 'merge1'

# In[37]:


# Inner Join을 활용헤 동 이름이 없는 행 제거
merge1 = pd.merge(left= p_df_cp, right = d_order, how = "inner", on = "법정동코드")
merge1.drop('읍면동명_y', axis=1, inplace=True)


# ### merge1과 경제활동인구수, 직장인구수 MERGE -> 'merge2'

# In[38]:


merge2 = pd.merge(left= merge1, right = commu_merge, how = "inner", left_on = "법정동코드", right_on = "법정동자택주소")
merge2.drop('법정동자택주소', axis=1, inplace=True)


# ### merge2와 성별 연령별 평균 연소득 금액, 상위소득자평균연소득금액, 중위연소득금액 MERGE -> 'merge3'

# In[39]:


merge3 = pd.merge(left= merge2, right = income_emd_group, how = "inner", left_on = "법정동코드", right_on = "읍면동코드")


# ### merge3와 성별 연령별 평균3개월카드이용금액 MERGE -> 'merge4'

# In[40]:


merge4 = pd.merge(left= merge3, right = spnd_emd_group, how = "inner", left_on = "법정동코드", right_on = "읍면동코드")
merge4.drop(['집계인구수','경제활동인구수'],axis=1, inplace=True)


# ### merge4와 유동인구 데이터  MERGE -> 'seoul'

# In[41]:


seoul = pd.merge(left= merge4, right = filtered_f_pop, how = "inner", left_on = "법정동코드", right_on = "law_code")
seoul = seoul.drop(['law_code'],axis=1)


# In[42]:


seoul_cp = seoul.copy()


# # 군집분석 (Elbow Method)

# In[43]:


from sklearn.preprocessing import MinMaxScaler
scale_seoul = seoul_cp.drop(['법정동코드','읍면동명_x'],axis=1)
# 비율이어서 스케일링x
x = scale_seoul.drop(['10_m_r','20_m_r','30_m_r','40_m_r','50_m_r','60_m_r','10_f_r','20_f_r','30_f_r','40_f_r','50_f_r','60_f_r',
                      '0시 유동인구','1시 유동인구','2시 유동인구','3시 유동인구','4시 유동인구','5시 유동인구','6시 유동인구','7시 유동인구'
                     ,'8시 유동인구','9시 유동인구','10시 유동인구','11시 유동인구','12시 유동인구','13시 유동인구','14시 유동인구'
                     ,'15시 유동인구','16시 유동인구','17시 유동인구','18시 유동인구','19시 유동인구','20시 유동인구','21시 유동인구'
                     ,'22시 유동인구','23시 유동인구'],axis=1)
# 비율이 아니어서 스케일링
x_s = scale_seoul[['10_m_r','20_m_r','30_m_r','40_m_r','50_m_r','60_m_r','10_f_r','20_f_r','30_f_r','40_f_r','50_f_r','60_f_r',
]]
x_t = scale_seoul[['0시 유동인구','1시 유동인구','2시 유동인구','3시 유동인구','4시 유동인구','5시 유동인구','6시 유동인구','7시 유동인구'
                     ,'8시 유동인구','9시 유동인구','10시 유동인구','11시 유동인구','12시 유동인구','13시 유동인구','14시 유동인구'
                     ,'15시 유동인구','16시 유동인구','17시 유동인구','18시 유동인구','19시 유동인구','20시 유동인구','21시 유동인구'
                     ,'22시 유동인구','23시 유동인구']]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
feature = pd.DataFrame(x_scaled)
col_name = scale_seoul.columns[12:64]
feature.columns = col_name
feature = pd.concat([x_s,feature],axis = 1)
feature = pd.concat([feature,x_t],axis = 1)


# In[44]:


feature_s = feature.fillna('0')
# 43번인덱스 false 값 발견.


# In[45]:


from sklearn.cluster import KMeans
import seaborn as sns

ks = range(2,19)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(feature_s)
    inertias.append(model.inertia_)
    
plt.figure(figsize=(20,10))
plt.plot(ks, inertias, '-o')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(ks)
plt.show()


# In[46]:


clust_model = KMeans(n_clusters = 11, n_init = 30, max_iter = 500, random_state= 34)
clust_model.fit(feature_s)
centers = clust_model.cluster_centers_
pred = clust_model.predict(feature_s)


# In[48]:


clust_df = feature_s.copy()
clust_df['clust'] = pred
clust_df.loc[:,'10_f_r':'60_f_r']=clust_df.loc[:,'10_f_r':'60_f_r'].astype(float)


# In[49]:


clust_df['clust'].value_counts()
a = clust_df.groupby('clust').mean()
a.rank(method = 'min', ascending = False).astype('int')


# ## 군집 별 전체 칼럼 시각화

# In[50]:


X = clust_df


# In[51]:


plt.rcParams["font.family"] = 'Gulim'
for i, col in enumerate(list(feature.columns)): 
    plt.figure(figsize=(50,50))
    plt.subplot(13,7,i+1)
    sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,i], data = feature, hue=clust_model.labels_, palette='coolwarm')
    plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,i], c='black', alpha = 0.8, s=30)
    plt.legend().remove()
    plt.show()


# ## 무의미한 칼럼(시각화)

# ### 1. 1/1/평균연소득금액 (15) 

# In[52]:


plt.rcParams["font.family"] = 'Gulim'
plt.figure(figsize=(50,50))
plt.subplot(13,5,1)
sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,15], data = feature, hue=clust_model.labels_, palette='coolwarm')
plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,15], c='black', alpha = 0.8, s=30)
plt.legend().remove()
plt.show()


# ### 2. 2/1/평균연소득금액

# In[53]:


plt.rcParams["font.family"] = 'Gulim'
plt.figure(figsize=(50,50))
plt.subplot(13,5,1)
sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,21], data = feature, hue=clust_model.labels_, palette='coolwarm')
plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,21], c='black', alpha = 0.8, s=10)
plt.legend().remove()
plt.show()


# ### 3. 1/1/상위소득자평균연소득금액

# In[54]:


plt.rcParams["font.family"] = 'Gulim'
plt.figure(figsize=(50,50))
plt.subplot(13,5,1)
sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,27], data = feature, hue=clust_model.labels_, palette='coolwarm')
plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,27], c='black', alpha = 0.8, s=10)
plt.legend().remove()
plt.show()


# ### 4. 2/1/상위소득자평균연소득금액

# In[55]:


plt.rcParams["font.family"] = 'Gulim'
plt.figure(figsize=(50,50))
plt.subplot(13,5,1)
sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,33], data = feature, hue=clust_model.labels_, palette='coolwarm')
plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,33], c='black', alpha = 0.8, s=10)
plt.legend().remove()
plt.show()


# ### 5. 1/1/중위연소득금액 

# In[56]:


plt.rcParams["font.family"] = 'Gulim'
plt.figure(figsize=(50,50))
plt.subplot(13,5,1)
sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,39], data = feature, hue=clust_model.labels_, palette='coolwarm')
plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,39], c='black', alpha = 0.8, s=10)
plt.legend().remove()
plt.show()


# ## 재 클러스터링

# In[57]:


from sklearn.preprocessing import MinMaxScaler
scale_seoul = seoul_cp.drop(['법정동코드','읍면동명_x'],axis=1)
# 비율이어서 스케일링x
x = scale_seoul.drop(['10_m_r','20_m_r','30_m_r','40_m_r','50_m_r','60_m_r','10_f_r','20_f_r','30_f_r','40_f_r','50_f_r','60_f_r',
                      '0시 유동인구','1시 유동인구','2시 유동인구','3시 유동인구','4시 유동인구','5시 유동인구','6시 유동인구','7시 유동인구'
                     ,'8시 유동인구','9시 유동인구','10시 유동인구','11시 유동인구','12시 유동인구','13시 유동인구','14시 유동인구'
                     ,'15시 유동인구','16시 유동인구','17시 유동인구','18시 유동인구','19시 유동인구','20시 유동인구','21시 유동인구'
                     ,'22시 유동인구','23시 유동인구'],axis=1)
# 비율이 아니어서 스케일링
x_s = scale_seoul[['10_m_r','20_m_r','30_m_r','40_m_r','50_m_r','60_m_r','10_f_r','20_f_r','30_f_r','40_f_r','50_f_r','60_f_r',
]]
x_t = scale_seoul[['0시 유동인구','1시 유동인구','2시 유동인구','3시 유동인구','4시 유동인구','5시 유동인구','6시 유동인구','7시 유동인구'
                     ,'8시 유동인구','9시 유동인구','10시 유동인구','11시 유동인구','12시 유동인구','13시 유동인구','14시 유동인구'
                     ,'15시 유동인구','16시 유동인구','17시 유동인구','18시 유동인구','19시 유동인구','20시 유동인구','21시 유동인구'
                     ,'22시 유동인구','23시 유동인구']]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
feature = pd.DataFrame(x_scaled)
col_name = scale_seoul.columns[12:64]
feature.columns = col_name
feature = pd.concat([x_s,feature],axis = 1)
feature = pd.concat([feature,x_t],axis = 1)


# ### 무의미한 칼럼 제거

# In[58]:


# 43번인덱스 false 값 발견.
feature_s = feature.fillna('0')
feature_s = feature_s.drop(['1/1/평균연소득금액','2/1/평균연소득금액','1/1/상위소득자평균연소득금액','2/1/상위소득자평균연소득금액','2/5/상위소득자평균연소득금액','2/6/상위소득자평균연소득금액','1/1/중위연소득금액'],axis=1)


# In[65]:


from sklearn.cluster import KMeans
import seaborn as sns

ks = range(2,20)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(feature_s)
    inertias.append(model.inertia_)
    
plt.figure(figsize=(20,10))
plt.plot(ks, inertias, '-o')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(ks)
plt.show()


# In[59]:


clust_model = KMeans(n_clusters = 11, n_init = 30, max_iter = 500, random_state= 34)
clust_model.fit(feature_s)
centers = clust_model.cluster_centers_
pred = clust_model.predict(feature_s)


# In[60]:


clust_df = feature_s.copy()
clust_df['clust'] = pred


# In[62]:


a=clust_df.columns[6:12]
clust_df[a]=clust_df[a].astype(float)


# In[63]:


X = clust_df


# In[64]:


plt.rcParams["font.family"] = 'Gulim'
for i, col in enumerate(list(feature.columns)): 
    plt.figure(figsize=(50,50))
    plt.subplot(13,9,i+1)
    sns.scatterplot(x = X.iloc[:,-1], y = X.iloc[:,i], data = feature, hue=clust_model.labels_, palette='coolwarm')
    plt.scatter([0,1,2,3,4,5,6,7,8,9,10], centers[:,i], c='black', alpha = 0.8, s=30)
    plt.legend().remove()
    plt.show()


# In[66]:


pd.set_option('display.max_columns', None)
cluster_mean = clust_df.groupby('clust').mean()


# In[67]:


cluster_mean.rank(method = 'min', ascending = False).astype('int')


# In[68]:


cluster_mean.sort_values(by="주문건수",ascending=False)


# In[69]:


inverse_seoul = pd.DataFrame()
for i in (clust_df.columns[12:57]):
    min_val = seoul_cp[i].min()
    max_val = seoul_cp[i].max()
    a = []
    for j in range(len(clust_df)):
        a.append(clust_df[i][j] * ( max_val - min_val ) + min_val)
    inverse_seoul[i] = a


# In[70]:


a = clust_df['clust']
a = pd.DataFrame(a)
col_names  = clust_df.columns
b = clust_df[col_names[0:12]]
c = clust_df[col_names[57:-1]]
d = seoul_cp[['법정동코드','읍면동명_x']]


# In[71]:


final_inverse_seoul = pd.concat([inverse_seoul, a], axis=1)
final_inverse_seoul = pd.concat([b,final_inverse_seoul],axis=1)
final_inverse_seoul = pd.concat([final_inverse_seoul, c],axis=1)
final_inverse_seoul = pd.concat([d,final_inverse_seoul],axis=1)


# In[72]:


pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_columns', None)
cluster_mean_invers = final_inverse_seoul.groupby('clust').mean()


# In[73]:


cluster_mean_invers.rank(method='min',ascending=False).sort_values(by="주문건수",ascending=True).astype('int')


# In[74]:


cluster_mean_invers.sort_values(by="주문건수",ascending=False)


# # 최종 클러스터링 결과

# In[75]:


result = pd.concat([clust_df['clust'],final_inverse_seoul],axis=1)


# In[76]:


#pd.set_option('display.max_rows', None)
#result
#다시 요약해서 보여주는 설정
#pd.options.display.max_rows=60
result.to_csv('result.csv')


# ## 1.클러스터 별 유동인구 비교 시각화

# In[77]:


vis_fp = pd.DataFrame()

for i in range(0,24):
               
    vis_fp['%s시 유동인구' %i] = cluster_mean_invers['유동인구총합계'] * cluster_mean_invers['%s시 유동인구' %i]
    
vis_fp.reset_index(inplace=False)


# In[78]:


fp_list=[]

for i in range(len(vis_fp)):
    
    a = vis_fp.iloc[i,:] 
    
    fp_list.append(a)


# In[79]:


vis_fp_cols=vis_fp.columns[:]


# In[80]:


from matplotlib import pyplot as plt

# 숫자랑 색 지정해서 
#color_list=['b','g','r','c','m','y','k','w','limegreen','violet','dodgerblue','orange']

number=range(10)

plt.figure(figsize=(40,20))
# 갂각 라벨로 넣어서 한개씩 그래프 그리기.
x_value= vis_fp_cols

for i, number in zip(fp_list, number):
    
    y_value_0 = i
    
    plt.plot(x_value, y_value_0, label=number)
    
plt.legend(loc='best')

plt.show()


# ## 2.클러스터별 주문건수 비교 시각화

# In[81]:


plt.bar(cluster_mean_invers.index,cluster_mean_invers['주문건수'])


# ### 3. 중위소득 시각화

# In[85]:


vis_mid_income = cluster_mean_invers.iloc[:,34:45]
vis_mid_income = vis_mid_income.drop(['2/1/중위연소득금액'],axis=1)


# In[86]:


vis_mid_income.sum(axis=1)


# In[87]:


y_mid_income=[]
for i in range(11):
    a = vis_mid_income.iloc[i,:] 
    y_mid_income.append(a)
y_mid_income


# In[88]:


from matplotlib import pyplot as plt
# 숫자랑 색 지정해서 
color_list=['b','g','r','c','m','y','k','w','limegreen','violet','dodgerblue','orange']
number=range(11)

plt.figure(figsize=(20,20))
# 갂각 라벨로 넣어서 한개씩 그래프 그리기.
x_value=vis_mid_income.columns[:]
for i,col,number in zip(y_mid_income,color_list,number):
    y_value_0=i
    plt.plot(x_value,y_value_0,color=col,label=number)
plt.legend(loc='best')
plt.show()


# ### 4. 소비데이터 시각화

# In[89]:


vis_spnd = cluster_mean_invers.iloc[:,45:57]
vis_spnd.columns = ['1/1','1/2','1/3','1/4','1/5','1/6','2/1','2/2','2/3','2/4','2/5','2/6']


# In[90]:


vis_spnd.sum(axis=1)


# In[91]:


y_spnd=[]
for i in range(11):
    a = vis_spnd.iloc[i,:]
    y_spnd.append(a)


# In[92]:


from matplotlib import pyplot as plt
# 숫자랑 색 지정해서 
color_list=['b','g','r','c','m','y','k','w','limegreen','violet','dodgerblue','orange']
number=range(11)

plt.figure(figsize=(20,20))
# 갂각 라벨로 넣어서 한개씩 그래프 그리기.
x_value=vis_spnd.columns[:]
for i,col,number in zip(y_spnd,color_list,number):
    y_value_0=i
    plt.plot(x_value,y_value_0,color=col,label=number)
plt.legend(loc='best')
plt.show()


# ### 5. 성비 시각화

# In[93]:


vis_m_ratio = cluster_mean_invers.iloc[:,1:6].sum(axis=1)
vis_f_ratio = cluster_mean_invers.iloc[:,7:13].sum(axis=1)


# In[94]:


plt.bar(vis_m_ratio.index,vis_m_ratio)
plt.show()


# In[95]:


plt.bar(vis_f_ratio.index,vis_f_ratio)
plt.show()

