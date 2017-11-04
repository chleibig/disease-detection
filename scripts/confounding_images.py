import cPickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()
sns.set_context('paper', font_scale=1.4)
A4_WIDTH = 8.27

with open('data/processed/'
          '100_mc_KaggleDR_test_BayesJFnet17_392bea6.pkl', 'rb') as h:
    pred_kaggle_1 = pickle.load(h)
with open('data/processed/'
          '100_mc_imagenet_val_BayesJFnet17_392bea6.pkl', 'rb') as h:
    pred_imagenet_1 = pickle.load(h)
with open('data/processed/'
          '100_mc_KaggleDR_test_bcnn2_b69aadd.pkl', 'rb') as h:
    pred_kaggle_2 = pickle.load(h)
with open('data/processed/'
          '100_mc_imagenet_val_BayesianJFnet17_onset2_b69aadd.pkl', 'rb') as h:
    pred_imagenet_2 = pickle.load(h)

pred_std_kaggle_1 = pred_kaggle_1['stoch_out'].std(axis=-1)[:, 1]
pred_std_imagenet_1 = pred_imagenet_1['stoch_out'].std(axis=-1)[:, 1]
pred_std_kaggle_2 = pred_kaggle_2['stoch_out'].std(axis=-1)[:, 1]
pred_std_imagenet_2 = pred_imagenet_2['stoch_out'].std(axis=-1)[:, 1]

plt.figure(figsize=(A4_WIDTH, A4_WIDTH // 2))

plt.subplot(121)
plt.title('(a) DNN for disease onset 1')
sns.kdeplot(pred_std_kaggle_1, shade=True, label='FUNDUS images')
sns.kdeplot(pred_std_imagenet_1, shade=True, label='non-FUNDUS images')
plt.xlabel('model uncertainty')
plt.ylabel('density')
plt.legend(loc='upper right')

plt.subplot(122)
plt.title('(b) DNN for disease onset 2')
sns.kdeplot(pred_std_kaggle_2, shade=True, label='FUNDUS images')
sns.kdeplot(pred_std_imagenet_2, shade=True, label='non-FUNDUS images')
plt.xlabel('model uncertainty')
plt.legend(loc='upper right')
