import pandas as pd
import numpy as np
import re
from tabulate import tabulate

df = pd.read_csv(r"C:\Users\13298\Desktop\invest_capitalflow.csv", low_memory=False)
df.columns = df.columns.str.lower()

def clean_amount(series):

    cleaned = (
        series
        .astype(str)
        .str.strip()  
        .str.upper() 

        .replace({
            'N/A': np.nan,
            'NULL': np.nan,
            '.': np.nan,
            '—': np.nan
        })
    )
    
    patterns = [
        (r'^([+-]?\d+\.?\d*)(万|千)$', 
         lambda m: str(float(m.group(1)) * 10000 if m.group(2)=="万" else float(m.group(1))*1000)),
        (r'^([+-]?\d+\.?\d*)[Ee][+-]?(\d+)$', 
         lambda m: str(float(m.group(1)) * 10**int(m.group(2)))),
        (r',', ''),
        (r'[^\d.-]', '')
    ]
    
    for pattern, repl in patterns:
        cleaned = cleaned.str.replace(pattern, repl, regex=True)
    
    def validate(x):
        try:
            if pd.isna(x) or x in ['', '.']:
                return np.nan
            
            parts = x.split('.')
            if len(parts) > 2:  
                return np.nan
            
            if '-' in x:
                if x.count('-') > 1 or not x.startswith('-'):
                    return np.nan
                
            value = float(x)
            
            return abs(value) if value < 0 else value
        except:
            return np.nan

    cleaned = cleaned.apply(validate)
    
    return cleaned

df['amount'] = (
    clean_amount(df['amount'])
    .fillna(0)  
    .round(2)   
)

original_count = len(df)  

high_amount_threshold = 1_000_0.00

df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

df = df.loc[
    (df['amount'] <= high_amount_threshold) | 
    (df['amount'].isna())                     
].copy()

df['amount'] = df['amount'].fillna(0)

print("\n金额列清洗报告：")
print(f"原始记录数：{original_count}")
print(f"保留记录数：{len(df)}（删除{original_count - len(df)}条，含高额过滤）")  
print(f"有效转换数：{df['amount'].notnull().sum()} ({df['amount'].notnull().mean():.2%})")
print(f"填充缺失值：{(df['amount'] == 0).sum()} ({df['amount'].eq(0).mean():.2%})")
print(f"数值分布统计：\n{df['amount'].describe()}")
print(f"负值检测：{df[df['amount'] < 0].shape[0]} 条")

current_max = df['amount'].max()
print(f"\n验证结果：当前最大金额 {current_max:.2f}")

assert df['amount'].min() >= 0, "存在负值未处理！"
assert df['amount'].notnull().all(), "存在未处理的缺失值！"

mask_dict = {
    "完整数据": df[['city_code_branch', 'city_code_sh']].notnull().all(axis=1),
    "双重缺失": df[['city_code_branch', 'city_code_sh']].isnull().all(axis=1),
    "仅分支缺失": df['city_code_branch'].isnull() & df['city_code_sh'].notnull(),
    "仅总部缺失": df['city_code_sh'].isnull() & df['city_code_branch'].notnull()
}

only_head_missing_company_names = df[mask_dict["仅总部缺失"]]['company_name_sh']
print(only_head_missing_company_names)
#only_head_missing_company_names.to_csv(r"C:\Users\13298\Desktop\only_head_missing_company_names.csv", index=False, encoding='utf-8-sig')

for k, v in mask_dict.items():
    print(f"{k}比例：{v.sum()/len(df):.2%}")

df['year'] = pd.to_datetime(df['establish_time_branch']).dt.year

geo = pd.read_csv(r"C:\Users\13298\Desktop\Citycode_1990_2019.csv")
geo['city_code'] = geo['city_code'].astype(str)

county_geo = pd.read_csv(r"C:\Users\13298\Desktop\tidy_countycode_1990_2019.csv")
county_geo = county_geo[['code', 'year', 'citychn']].drop_duplicates()
county_geo['code'] = county_geo['code'].astype(str)

df['city_code_sh'] = df['city_code_sh'].fillna(0).astype(int).astype(str).replace('0', np.nan)
df['city_code_branch'] = df['city_code_branch'].fillna(0).astype(int).astype(str).replace('0', np.nan)

def enhanced_merge(df, code_col, geo_df, county_df, city_field):
    df[code_col] = df[code_col].astype('category')
    geo_df['city_code'] = geo_df['city_code'].astype('category')
    county_df['code'] = county_df['code'].astype('category')
    
    geo_map = geo_df.set_index(['city_code', 'year'])['citychn'].to_dict()
    county_map = county_df.set_index(['code', 'year'])['citychn'].to_dict()
    
    keys = pd.Series(zip(df[code_col], df['year']), index=df.index)
    
    merged = df.assign(**{city_field: keys.map(geo_map)})
    
    county_mask = merged[city_field].isna()

    valid_mask = county_mask & (keys.index.isin(merged.index))
    merged.loc[valid_mask, city_field] = keys[valid_mask].map(county_map)
    
    max_year_per_code = geo_df.groupby('city_code')['year'].max().to_dict()
    valid_years = df[code_col].map(max_year_per_code).fillna(df['year'])
    adjusted_years = np.where(df['year'] > valid_years, valid_years, df['year'])
    
    adjusted_keys = pd.Series(zip(df[code_col], adjusted_years), index=df.index)
    fallback_match = adjusted_keys.map(geo_map)
    
    merged[city_field] = merged[city_field].fillna(fallback_match)
    
    special_codes = {
        '110100': '北京市',
        '110200': '北京市',
        '110300': '北京市',
        '310100': '上海市',
        '310200': '上海市',
        '120200': '天津市',
        '120100': '天津市',
        '500100': '重庆市',
        '500200': '重庆市',
        '500300': '重庆市',
        '500900': '重庆市',
        '440100': '广州市',
        '330100': '杭州市',
        '330300': '温州市',
        '220100': '长春市'
    }

    merged[city_field] = np.select(
        [merged[code_col].isin(special_codes.keys())],
        [merged[code_col].map(special_codes)],
        default=merged[city_field]
    )
    
    return merged

merged = enhanced_merge(df, 'city_code_sh', geo, county_geo, 'citya')
merged = merged.rename(columns={'city_code': 'city_code_sh_merged'})

merged = enhanced_merge(merged, 'city_code_branch', geo, county_geo, 'cityb')
merged = merged.rename(columns={'city_code': 'city_code_branch_merged'})


failed_sh = merged['city_code_sh'].notnull() & merged['citya'].isnull()
failed_branch = merged['city_code_branch'].notnull() & merged['cityb'].isnull()

print("\n总部匹配失败分析：")
total_failed_sh = failed_sh.sum()
print(f"总匹配失败样本数：{total_failed_sh}")

sh_with_code = merged['city_code_sh'].notnull()
failed_with_code = merged.loc[failed_sh & sh_with_code]
failed_with_code_count = len(failed_with_code)
print(f"├─ 有代码匹配失败：{failed_with_code_count}（占比{failed_with_code_count/total_failed_sh:.2%}）")

valid_mask = failed_with_code['year'].between(1990, 2019)
valid_failed_sh = failed_with_code[valid_mask]

prov_code_failures_sh = valid_failed_sh['city_code_sh'].str[-4:].eq('0000').sum()
valid_failed_count = len(valid_failed_sh)

print(f"│   ├─ 有效年份内（1990-2019）：{valid_failed_count}（占此类失败的{valid_failed_count/failed_with_code_count:.2%}）")
print(f"│   │   ├─ 省级代码导致失败：{prov_code_failures_sh}（占有效年份失败的{prov_code_failures_sh/valid_failed_count:.2%}）")
print(f"│   │   └─ 其他原因失败：{valid_failed_count-prov_code_failures_sh}（占有效年份失败的{(valid_failed_count-prov_code_failures_sh)/valid_failed_count:.2%}）")
print(f"│   └─ 其他年份：{failed_with_code_count - valid_failed_count}（占此类失败的{(failed_with_code_count - valid_failed_count)/failed_with_code_count:.2%}）")

failed_sh_samples = merged.loc[
    failed_sh & 
    merged['year'].between(1990, 2019) &
    ~merged['city_code_sh'].str.endswith('0000', na=False), 
    [
    'company_id',        
    'invest_company_id',   
    'company_name_sh',     
    'company_name_branch', 
    'city_code_sh',        
    'year',                
    'amount',              
    'category_code_sh'     
]].dropna(subset=['city_code_sh'])  

failed_sh_samples.to_csv(r"C:\Users\13298\Desktop\failed_sh_codes.csv", 
                        index=False,
                        encoding='utf-8-sig')

print("\n分支机构匹配失败分析：")
total_failed_branch = failed_branch.sum()
print(f"总匹配失败样本数：{total_failed_branch}")

branch_with_code = merged['city_code_branch'].notnull()
failed_branch_with_code = merged.loc[failed_branch & branch_with_code]
failed_branch_with_code_count = len(failed_branch_with_code)
print(f"├─ 有代码匹配失败：{failed_branch_with_code_count}（占比{failed_branch_with_code_count/total_failed_branch:.2%}）")

valid_mask_branch = failed_branch_with_code['year'].between(1990, 2019)
valid_failed_branch = failed_branch_with_code[valid_mask_branch]

prov_code_failures_branch = valid_failed_branch['city_code_branch'].str[-4:].eq('0000').sum()
valid_failed_branch_count = len(valid_failed_branch)

print(f"│   ├─ 有效年份内（1990-2019）：{valid_failed_branch_count}（占此类失败的{valid_failed_branch_count/failed_branch_with_code_count:.2%}）")
print(f"│   │   ├─ 省级代码导致失败：{prov_code_failures_branch}（占有效年份失败的{prov_code_failures_branch/valid_failed_branch_count:.2%}）")
print(f"│   │   └─ 其他原因失败：{valid_failed_branch_count-prov_code_failures_branch}（占有效年份失败的{(valid_failed_branch_count-prov_code_failures_branch)/valid_failed_branch_count:.2%}）")
print(f"│   └─ 其他年份：{failed_branch_with_code_count - valid_failed_branch_count}（占此类失败的{(failed_branch_with_code_count - valid_failed_branch_count)/failed_branch_with_code_count:.2%}）")

failed_branch_samples = merged.loc[
    failed_branch & 
    merged['year'].between(1990, 2019) &
    ~merged['city_code_branch'].str.endswith('0000', na=False), 
    [
    'company_id',          
    'invest_company_id',   
    'company_name_branch', 
    'company_name_sh',    
    'city_code_branch',    
    'year',                
    'amount',              
    'category_code_branch' 
]].dropna(subset=['city_code_branch'])  

failed_branch_samples.to_csv(r"C:\Users\13298\Desktop\failed_branch_codes.csv", 
                            index=False,
                            encoding='utf-8-sig')

top_amount_samples = (
    merged
    .nlargest(50, 'amount')
    [['year', 
      'company_name_sh', 
      'company_name_branch', 
      'amount', 
      'city_code_sh', 
      'city_code_branch']]
    .assign(amount=lambda x: x['amount'].apply(lambda v: f"{v:,.2f}万元"))
)

print("\n金额最高的50笔交易样本：")
print(tabulate(top_amount_samples, 
              headers='keys', 
              tablefmt='psql',
              showindex=False,
              numalign='right',
              stralign='left'))

output = (
    merged
    .loc[lambda x: x[['citya', 'cityb']].notnull().all(axis=1)]
    .assign(amount=pd.to_numeric(merged['amount'], errors='coerce').fillna(0))
    
    .groupby(['year', 'citya', 'cityb'], as_index=False)
    .agg(
        amount=('amount', 'sum'),
        amount_time=('amount', lambda s: (s != 0).sum())  
    )
    .assign(amount_unit='万元')
)

output.to_csv(r"C:\Users\13298\Desktop\city_investment_flow.csv", 
             index=False,
             encoding='utf-8-sig')

print(f"\n已生成聚合数据，共{len(output)}条记录")

city_investment_flow_path = r"C:\Users\13298\Desktop\city_investment_flow.csv"
merged_dataset_path = r"C:\Users\13298\Desktop\panel_data\merged_dataset.csv"
output_path = r"C:\Users\13298\Desktop\merged_dataset_with_amount.csv"

try:
    city_investment_flow = pd.read_csv(city_investment_flow_path)
    merged_dataset = pd.read_csv(merged_dataset_path)
    print(f"投资流数据形状：{city_investment_flow.shape}，面板数据形状：{merged_dataset.shape}")
    # 统一列名并过滤所需列
    flow_df = city_investment_flow.rename(columns={
        'citya': 'city_a',
        'cityb': 'city_b'
    })[['city_a', 'city_b', 'year', 'amount', 'amount_time']]

    merged_dataset = pd.merge(
        merged_dataset,
        flow_df,
        on=['city_a', 'city_b', 'year'],
        how='left'
    ).fillna({'amount': 0, 'amount_time': 0})

    merged_dataset.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"已成功保存至 {output_path}，记录数：{len(merged_dataset)}")

except FileNotFoundError:
    print("未找到指定的文件")
except Exception as e:
    print(f"发生未知错误：{e}")

file = pd.read_csv(output_path)

statistic=pd.DataFrame(file)

grouped_amount = statistic.groupby('connect')['amount'].agg(['mean', 'median', 'min', 'max', 'std'])

print("connect 为 0 和 1 时 amount 列的统计信息：")
print(grouped_amount)

grouped_time = statistic.groupby('connect')['amount_time'].agg(['mean', 'median', 'min', 'max', 'std'])

print("connect 为 0 和 1 时 amount_time 列的统计信息：")
print(grouped_time)