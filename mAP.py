"""
compute mAP given a result file
"""
import glob, os
import re
import numpy as np
import  matplotlib.pyplot as plt
regex = ''

def get_map(file):
    mAP = 0.0
    i = 0

    with open(file) as f:        
        for line in f :
            result = line.strip().split(',')
            ranking = np.array([int(int(elem) == int(result[0])) for elem in result[1:]])
            inds = np.arange(1, len(ranking)+1)
            recall = np.cumsum(ranking) * ranking
            valid_inds = recall != 0
            inds = inds[valid_inds]
            recall = recall[valid_inds]
            precision = recall / inds
            AP = np.mean(precision)
            if np.isnan(AP): AP = 0.0 
            mAP += AP
            i +=1
    return mAP/i
            

if __name__ == '__main__':
    folder = os.path.join(os.getcwd(), 'results/')
    dmAP = {}
    names = []
    original_names = []
    colors = []
    print(os.listdir(folder))
    for idx, file in enumerate(sorted(glob.glob(os.path.join(folder, "*.csv")))):
        print(file)
        name_file = file.split('/')[-1]
        original_names.append(name_file)
        names.append(name_file.split("_")[0].capitalize() + " encoder \n" + name_file.split("_")[-1].split(".")[0].capitalize())
        dmAP[idx] = get_map(file)

        if name_file.startswith("two"):
            colors.append('red')
        elif name_file.startswith("one"):
            colors.append('blue')
        else:
            colors.append('gray') 

    keys = list(dmAP.keys())
    keys = sorted(keys)
    its = []
    mAPs = []
    avg_map = 0.0

    for it in keys:
        print('{}: {}'.format(original_names[it], dmAP[it]), flush=True)
        its.append(it)
        mAPs.append(dmAP[it])
        avg_map += dmAP[it]

    print('final: {}'.format(avg_map/len(mAPs)), flush=True)
    plt.bar(names, mAPs, color=colors)
    plt.bar(0.5, 0, color='white', width=0.5)
    plt.title('mAP')

    for i, mAP in enumerate(mAPs):
        plt.text(i, mAP, f'{mAP:.2f}', ha='center', va='bottom')
    plt.show()

# ----------------------- all -----------------------
# model_paper_sbir_vit16_quickdrawext_p1.csv: 0.07729095930802402
# model_paper_sbir_vit16_quickdrawext_p10.csv: 0.05686089861284299
# model_paper_sbir_vit16_quickdrawext_p11.csv: 0.15588229862581113
# model_paper_sbir_vit16_quickdrawext_p12.csv: 0.38603180346384475
# model_paper_sbir_vit16_quickdrawext_p13.csv: 0.11953598851323889
# model_paper_sbir_vit16_quickdrawext_p14.csv: 0.15090949036552595
# model_paper_sbir_vit16_quickdrawext_p2.csv: 0.14326497005629293
# model_paper_sbir_vit16_quickdrawext_p3.csv: 0.2191023749809806
# model_paper_sbir_vit16_quickdrawext_p4.csv: 0.1537067363885815
# model_paper_sbir_vit16_quickdrawext_p5.csv: 0.11809513533697591
# model_paper_sbir_vit16_quickdrawext_p6.csv: 0.08221962312496563
# model_paper_sbir_vit16_quickdrawext_p7.csv: 0.06524443492890536
# model_paper_sbir_vit16_quickdrawext_p8.csv: 0.08093899148590258
# model_paper_sbir_vit16_quickdrawext_p9.csv: 0.13184510086208148
# final: 0.13863777186099815

# ----------------------- @200 -----------------------
# model_paper_sbir_vit16_quickdrawext_p1.csv: 0.17892819852499775
# model_paper_sbir_vit16_quickdrawext_p10.csv: 0.07513200400834523
# model_paper_sbir_vit16_quickdrawext_p11.csv: 0.17599475395986258
# model_paper_sbir_vit16_quickdrawext_p12.csv: 0.40026382881715794
# model_paper_sbir_vit16_quickdrawext_p13.csv: 0.33342505025799396
# model_paper_sbir_vit16_quickdrawext_p14.csv: 0.4969261467097327
# model_paper_sbir_vit16_quickdrawext_p2.csv: 0.3884919703014226
# model_paper_sbir_vit16_quickdrawext_p3.csv: 0.5375886054406644
# model_paper_sbir_vit16_quickdrawext_p4.csv: 0.41830676087233815
# model_paper_sbir_vit16_quickdrawext_p5.csv: 0.30860461801297256
# model_paper_sbir_vit16_quickdrawext_p6.csv: 0.20303166547609683
# model_paper_sbir_vit16_quickdrawext_p7.csv: 0.13827428482562001
# model_paper_sbir_vit16_quickdrawext_p8.csv: 0.2190508152354461
# model_paper_sbir_vit16_quickdrawext_p9.csv: 0.40573404106579647
# final: 0.2941745403945789

#
# python -m experiments.LN_prompt --exp_name=LN_prompt --n_prompts=3 --clip_LN_lr=1e-5 --prompt_lr=3e-4 --batch_size=16 --workers=12 --model_type=one_encoder --max_size=224 --fg 


# DINO @all
# clip_vit16_qdext_1.csv: 0.14088817969564196
# clip_vit16_qdext_10.csv: 0.0668894718505015
# clip_vit16_qdext_11.csv: 0.10882222532714643
# clip_vit16_qdext_12.csv: 0.15028941466746448
# clip_vit16_qdext_13.csv: 0.2549125300510566
# clip_vit16_qdext_14.csv: 0.31911563082354283
# clip_vit16_qdext_2.csv: 0.16236736910779423
# clip_vit16_qdext_3.csv: 0.3267704713524469
# clip_vit16_qdext_4.csv: 0.25570794673985686
# clip_vit16_qdext_5.csv: 0.18779265159323263
# clip_vit16_qdext_6.csv: 0.11111687293503035
# clip_vit16_qdext_7.csv: 0.06379675430946738
# clip_vit16_qdext_8.csv: 0.10664737802205179
# clip_vit16_qdext_9.csv: 0.2059947163543879
# final: 0.17579368663068726

# DINO @200
# clip_vit16_qdext_1.csv: 0.24110268783964434
# clip_vit16_qdext_10.csv: 0.07822349976790775
# clip_vit16_qdext_11.csv: 0.17612603962929346
# clip_vit16_qdext_12.csv: 0.15616009862838992
# clip_vit16_qdext_13.csv: 0.4576207173418288
# clip_vit16_qdext_14.csv: 0.567120913370903
# clip_vit16_qdext_2.csv: 0.3039592832471211
# clip_vit16_qdext_3.csv: 0.5408466576242559
# clip_vit16_qdext_4.csv: 0.4965776110301784
# clip_vit16_qdext_5.csv: 0.33085429093220725
# clip_vit16_qdext_6.csv: 0.2296023489161038
# clip_vit16_qdext_7.csv: 0.0899699663434947
# clip_vit16_qdext_8.csv: 0.16399773824117816
# clip_vit16_qdext_9.csv: 0.510664335458907
# final: 0.31020187059795806
