
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return accuracy, auc, specificity, sensitivity, f1

# Function to calculate the ROC curve
def getROC_input(df):
    label,predict = np.array(df["Label"]),np.array(df["Prob"])
    # print( label,predict)
    fpr, tpr, _ = roc_curve(label, predict, pos_label=1)
    # print(fpr,tpr)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_metrics_df(df, datasets='dataset', label_col='Label', prob_col='Prob'):
    metrics = {'ACC': [], 'AUC': [], 'Spe.': [], 'Sen.': [], 'F1': []}
    # datasets = df[dataset_col].unique()
    for dataset in datasets:
        print(dataset)
        accuracies = []
        aucs = []
        specificities = []
        sensitivities = []
        f1_scores = []
        for fold in df['Fold'].unique():
            fold_data = df[(df['dataset'] == dataset) & (df['Fold'] == fold)]
            if fold_data.shape[0] == 0:
                continue
            label = fold_data[label_col]
            prob = fold_data[prob_col] > 0.5
            accuracies.append(accuracy_score(label, prob))
            aucs.append(roc_auc_score(label, fold_data[prob_col]))
            specificities.append(recall_score(label, prob, pos_label=0))
            sensitivities.append(recall_score(label, prob, pos_label=1))
            f1_scores.append(f1_score(label, prob))
        
        metrics['ACC'].append((np.mean(accuracies), np.std(accuracies)))
        print(f" ACC: {np.mean(accuracies):.3f}±{np.std(accuracies):.3f}")
        metrics['AUC'].append((np.mean(aucs), np.std(aucs)))
        print(f" AUC: {np.mean(aucs):.3f}±{np.std(aucs):.3f}")
        metrics['Spe.'].append((np.mean(specificities), np.std(specificities)))
        print(f" Spe.: {np.mean(specificities):.3f}±{np.std(specificities):.3f}")
        metrics['Sen.'].append((np.mean(sensitivities), np.std(sensitivities)))
        print(f" Sen.: {np.mean(sensitivities):.3f}±{np.std(sensitivities):.3f}")
        metrics['F1'].append((np.mean(f1_scores), np.std(f1_scores)))
        print(f"F1: {np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}")
    
    metrics_df = pd.DataFrame(metrics, index=datasets)
    return metrics_df

# Function to plot the ROC curve with confidence intervals
def plt_combined_roc_with_confidence_interval(df,ax,genetype, datasets='dataset',plot_eachfold = False,legend_remove= False):
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams["axes.labelsize"] = 18
    palette = ["#F27970", '#BB9727', '#54B345', '#32B897', "#05B9E2", "#8983BF", "#C76DA2", "#5F97D2", "#9DC3E7"]
    
    lw = 2
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    

    for i, dataset in enumerate(datasets):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for j,fold in enumerate (df['Fold'].unique()):
            fold_data = df[(df['dataset'] == dataset) & (df['Fold'] == fold)]
            # print(fold_data)
            if fold_data.shape[0] == 0:
                continue
            fpr, tpr, roc_auc = getROC_input(fold_data)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
            if plot_eachfold:
                ax.plot(fpr, tpr, color=palette[j],
                        lw=lw, label='Fold{}'.format(j)+ '(AUROC = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

        mean_tpr = np.mean(tprs, axis=0)
        
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=palette[i % len(palette)],
                label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
                lw=lw, alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=palette[i % len(palette)], alpha=0.2)

    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'Receiver Operating Characteristic of {genetype} ', fontsize=14)
    if legend_remove:
        ax.legend().remove()
    else:
        ax.legend(loc="lower right",fontsize=12)
    # plt.show()
    return  ax

def plot_KM(df, title, y, HR=None):
    fig, ax =  plt.subplots(figsize=(6, 6), nrows=1, ncols=1) 
    lw = 2
    palette = ["#F27970", '#BB9727', '#54B345', '#32B897',
               "#05B9E2", "#8983BF", "#C76DA2", "#5F97D2", "#9DC3E7"]
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams["axes.labelsize"] = 18

    groups = df["Group"]
    
    labels = df["Group"].unique()
    kmf_list = []

    for i, label in enumerate(labels):
        ix = (groups == label)
        kmf = KaplanMeierFitter()
        kmf.fit(df['Time'][ix], df['Event'][ix], label=label)
        kmf_list.append(kmf)

        surv = kmf.survival_function_
        surv["timeline"] = surv.index
        # print(surv)

        ax.plot(surv["timeline"], surv[label], color=palette[i], lw=lw, label=label)

    # Log-rank test
    ix = (groups == labels[0])
    result = logrank_test(df['Time'][ix], df['Time'][~ix],
                          df['Event'][ix], df['Event'][~ix])

    # Plot decorations
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Relapse-Free survival (RFS)', fontsize=14)
    ax.legend(loc="upper right", fontsize=12, title=title,title_fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    # Add HR & p-value text
    x = 0.3
    if HR:
        ax.text(y, x, HR, fontsize=12, verticalalignment='center',
                horizontalalignment='left', color="k")
    if result.p_value < 0.001:
        ax.text(y, x - 0.1, "Log-rank test: P<0.001", fontsize=12,
                verticalalignment='center', horizontalalignment='left', color="k")
    else:
        ax.text(y, x - 0.1, "Log-rank test: P = {:.3f}".format(result.p_value),
                fontsize=12, verticalalignment='center', horizontalalignment='left', color="k")

 
    # ----------- Custom inline risk table (like 图2 style) ------------
    timepoints = [0, 500, 1000, 1500, 2000]
    y_start = 0  # risk table 起始 y 坐标
    y_step = -0.06   # 每行之间的间隔

    # # 横向时间标签
    
    ax.text(-300, -0.1, "Number at risk (%)", ha='right', va='top', fontsize=12)

    # 逐组添加 risk table 数字
    for i, kmf in enumerate(kmf_list):
        total = kmf.event_table.at_risk.iloc[0]
        for j, t in enumerate(timepoints):
            surv_prob = kmf.predict(t).values[0]
            num_alive = int(np.round(surv_prob * total))
            percent_alive = int(np.round(surv_prob * 100))
            text = f"{num_alive} ({percent_alive})"
            ax.text(t, y_start + y_step * (len(labels) - i +1.8), text,
                    ha='center', va='top', fontsize=12)

        # 添加组标签
        ax.text(-300, y_start + y_step * (len(labels) - i + 1.8), kmf._label, 
                ha='right', va='top', fontsize=12)

    # 调整主图以显示底部风险表
    ax.set_ylim(bottom=y_start - 0.05)

    plt.tight_layout()
    plt.show()