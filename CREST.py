import numpy as np 
import utls
import sklearn.linear_model as sk
import sklearn.discriminant_analysis as sklda
from sklearn.metrics import roc_auc_score


class CREST():

    def __init__(self, numFilt=3, dist_measure='riem',logreg_solver='liblinear'):
        # numFilt: number of CSP or CTP filters for each class 
        # dist_measure: distance measure either 'riem' or 'log-eucl'
        # logreg_solver: the solver for the logistic regression algorithm 

        self.numFilt = numFilt
        self.dist_measure = dist_measure
        self.logreg_solver = logreg_solver

    def train(self, train_data_1, train_data_2):

        self.csp_ctp = CSP_CTP(self.numFilt)
        train_CSTP_P = self.csp_ctp.train(train_data_1, train_data_2)

        self.drm_st = DRM_ST(dist_measure=self.dist_measure, logreg_solver=self.logreg_solver)
        train_DRMST_P = self.drm_st.train(train_data_1, train_data_2)


        n_trials_1 = np.size(train_data_1,1)
        n_trials_2 = np.size(train_data_2,1)


        # train a logistic regression to combine the scores
        X = np.concatenate((np.asmatrix(train_CSTP_P[:,0]).transpose(), np.asmatrix(train_DRMST_P[:,0]).transpose()), axis=1)#, np.asmatrix(train_WM_P[:,0]).transpose()], axis=1)


        y1 = np.zeros(n_trials_1)
        y2 = np.ones(n_trials_2)
        y = np.concatenate((y1, y2))

        self.clf = sk.LogisticRegression(solver=self.logreg_solver)
        self.clf.fit(X, y)


    def accuracy(self,test_data_1, test_data_2):
        # calculate the classification accuracy

        n_trials_1 = np.size(test_data_1, 1)
        n_trials_2 = np.size(test_data_1, 1)

        # find the scores for each classifier 
        test_CSTP_P1, test_CSTP_P2  =self.csp_ctp.test(test_data_1, test_data_2)
        test_DRMST_P1, test_DRMST_P2  =self.drm_st.test(test_data_1, test_data_2)

        # combined the classifier scores
        test_score_1 = np.concatenate([np.asmatrix(test_CSTP_P1[:,0]).transpose(), np.asmatrix(test_DRMST_P1[:,0]).transpose()], axis=1)
        test_score_2 = np.concatenate([np.asmatrix(test_CSTP_P2[:,0]).transpose(), np.asmatrix(test_DRMST_P2[:,0]).transpose()], axis=1)

        # test for accuracy
        vect1 = self.clf.predict(test_score_1)
        vect2 = self.clf.predict(test_score_2)

        tempp = np.sum(vect1 == 0) + np.sum(vect2 == 1)

        # since classes are balanced, calculate the overall accuracy
        class_rate = tempp / float(n_trials_1 + n_trials_2)

        return class_rate

    def auc(self,test_data_1, test_data_2):
        # calculate the AUC

        n_trials_1 = np.size(test_data_1, 1)
        n_trials_2 = np.size(test_data_1, 1)

        # find the scores for each classifier 
        test_CSTP_P1, test_CSTP_P2  =self.csp_ctp.test(test_data_1, test_data_2)
        test_DRMST_P1, test_DRMST_P2  =self.drm_st.test(test_data_1, test_data_2)

        # combined the classifier scores
        test_score_1 = np.concatenate([np.asmatrix(test_CSTP_P1[:,0]).transpose(), np.asmatrix(test_DRMST_P1[:,0]).transpose()], axis=1)
        test_score_2 = np.concatenate([np.asmatrix(test_CSTP_P2[:,0]).transpose(), np.asmatrix(test_DRMST_P2[:,0]).transpose()], axis=1)

        # test for accuracy
        test_P1 = self.clf.predict_proba(test_score_1)
        test_P2 = self.clf.predict_proba(test_score_2)
        y_scores = np.concatenate([np.asmatrix(test_P1[:,1]).transpose(), np.asmatrix(test_P2[:,1]).transpose()], axis=0)

        y1_true = np.zeros(n_trials_1)
        y2_true = np.ones(n_trials_2)
        y_true = np.concatenate((y1_true, y2_true))

        return  roc_auc_score(y_true, y_scores)


class CSP_CTP():

    def __init__(self, numFilt = 3):
        # numFilt is a positive integer indicating the number of filters for each class
        # 2<= numFilt <= half of the number of channels
        
        self.numFilt = numFilt
        

    def train(self, train_data_1, train_data_2):    
        [n_freqs, n_trials_1, n_chans, n_tsamps] = np.shape(train_data_1)
        n_trials_2 = np.size(train_data_2,1)

        self.v_csp = n_freqs*[None]
        self.v_ctp_g = n_freqs*[None]
        self.v_ctp_b = n_freqs*[None]

        step = 2* self.numFilt * 2 * self.numFilt
        train_feats_1 = np.zeros((n_trials_1, n_freqs * step))
        train_feats_2 = np.zeros((n_trials_2, n_freqs * step))
        

        for il in range(n_freqs):
            # for each frequency band...

            self.v_csp[il] = utls.calculate_cp(train_data_1[il, :, :, :], train_data_2[il, :, :, :] , 'space')

            # select filters for each good and bad class 
            f_g = self.v_csp[il][:, 0:self.numFilt]
            train_filt_1_csp_g = utls.apply_cp(train_data_1[il, :, :, :], f_g, 'space', numFilt=None)
            train_filt_2_csp_g = utls.apply_cp(train_data_2[il, :, :, :], f_g, 'space', numFilt=None)
            self.v_ctp_g[il] = utls.calculate_cp(train_filt_1_csp_g, train_filt_2_csp_g, 'time')

            f_b = self.v_csp[il][:, n_chans-self.numFilt:n_chans]
            train_filt_1_csp_b = utls.apply_cp(train_data_1[il, :, :, :], f_b, 'space', numFilt=None)
            train_filt_2_csp_b = utls.apply_cp(train_data_2[il, :, :, :], f_b, 'space', numFilt=None)
            self.v_ctp_b[il] = utls.calculate_cp(train_filt_1_csp_b, train_filt_2_csp_b, 'time')


            # extract the features
            train_filt_1_g = utls.apply_cp(train_filt_1_csp_g, self.v_ctp_g[il], 'time', numFilt=self.numFilt)
            train_filt_1_b = utls.apply_cp(train_filt_1_csp_b, self.v_ctp_b[il], 'time', numFilt=self.numFilt)
            train_filt_1 = np.concatenate((train_filt_1_g, train_filt_1_b), axis=2)
            train_feats_1[:, il * step : (il+1) * step] = np.reshape(np.asarray(train_filt_1), (n_trials_1, 2 * self.numFilt * 2 * self.numFilt))

            train_filt_2_g = utls.apply_cp(train_filt_2_csp_g, self.v_ctp_g[il], 'time', numFilt=self.numFilt)
            train_filt_2_b = utls.apply_cp(train_filt_2_csp_b, self.v_ctp_b[il], 'time', numFilt=self.numFilt)
            train_filt_2 = np.concatenate((train_filt_2_g, train_filt_2_b), axis=2)
            train_feats_2[:, il * step : (il+1) * step] = np.reshape(np.asarray(train_filt_2), (n_trials_2, 2 * self.numFilt * 2 * self.numFilt))

        self.clf = sklda.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        X = np.concatenate((train_feats_1, train_feats_2), axis=0)

        y1 = np.zeros(n_trials_1)
        y2 = np.ones(n_trials_2)
        y = np.concatenate((y1, y2))

        self.clf.fit(X, y)

        train_P = self.clf.predict_proba(X)

        return train_P

    def test(self, test_data_1, test_data_2):    
        [n_freqs, n_trials_1, n_chans, n_tsamps] = np.shape(test_data_1)
        n_trials_2 = np.size(test_data_2,1)

        step = 2* self.numFilt * 2 * self.numFilt
        test_logP_1 = np.zeros((n_trials_1, n_freqs * step))
        test_logP_2 = np.zeros((n_trials_2, n_freqs * step))


        for il in range(n_freqs):
            # for each frequency band...

            f_g = self.v_csp[il][:, 0:self.numFilt]
            test_filt_G_csp_g = utls.apply_cp(test_data_1[il, :, :, :], f_g, 'space')
            test_filt_B_csp_g = utls.apply_cp(test_data_2[il, :, :, :], f_g, 'space')

            f_b = self.v_csp[il][:, n_chans-self.numFilt:n_chans]
            test_filt_G_csp_b = utls.apply_cp(test_data_1[il, :, :, :], f_b, 'space')
            test_filt_B_csp_b = utls.apply_cp(test_data_2[il, :, :, :], f_b, 'space')

            # test the classifier on the test data
            test_filt_G_g = utls.apply_cp(test_filt_G_csp_g, self.v_ctp_g[il], 'time', numFilt=self.numFilt)
            test_filt_G_b = utls.apply_cp(test_filt_G_csp_b, self.v_ctp_b[il], 'time', numFilt=self.numFilt)
            test_filt_G = np.concatenate((test_filt_G_g, test_filt_G_b), axis=2)
            test_logP_1[:, il * step : (il+1) * step] = np.reshape(np.asarray(test_filt_G), (n_trials_1, 2 * self.numFilt * 2 * self.numFilt))

            test_filt_B_g = utls.apply_cp(test_filt_B_csp_g, self.v_ctp_g[il], 'time', numFilt=self.numFilt)
            test_filt_B_b = utls.apply_cp(test_filt_B_csp_b, self.v_ctp_b[il], 'time', numFilt=self.numFilt)
            test_filt_B = np.concatenate((test_filt_B_g, test_filt_B_b), axis=2)
            test_logP_2[:, il * step : (il+1) * step] = np.reshape(np.asarray(test_filt_B), (n_trials_2, 2 * self.numFilt * 2 * self.numFilt))

        test_P1 = self.clf.predict_proba(test_logP_1)
        test_P2 = self.clf.predict_proba(test_logP_2)

        return test_P1, test_P2

    def accuracy(self,test_data_1, test_data_2):
        # calculate the classification accuracy

        n_trials_1 = np.size(test_data_1, 1)
        n_trials_2 = np.size(test_data_1, 1)

        # find the scores for each classifier 
        test_P1, test_P2  =self.test(test_data_1, test_data_2)

        tempp = np.sum(test_P1[:,0]>0.5) + np.sum(test_P2[:,1]>0.5)

        # since classes are balanced, calculate the overall accuracy
        class_rate = tempp / float(n_trials_1 + n_trials_2)

        return class_rate

    def auc(self,test_data_1, test_data_2):
        # calculate the AUC

        n_trials_1 = np.size(test_data_1, 1)
        n_trials_2 = np.size(test_data_1, 1)

        # find the scores for each classifier 
        test_P1, test_P2  =self.test(test_data_1, test_data_2)
        y_scores = np.concatenate([np.asmatrix(test_P1[:,1]).transpose(), np.asmatrix(test_P2[:,1]).transpose()], axis=0)

        y1_true = np.zeros(n_trials_1)
        y2_true = np.ones(n_trials_2)
        y_true = np.concatenate((y1_true, y2_true))

        return  roc_auc_score(y_true, y_scores)



class DRM_ST():

    def __init__(self, dist_measure = 'riem',logreg_solver='liblinear'):
        self.rType = dist_measure
        self.logreg_solver = logreg_solver

    def train(self, train_data_1, train_data_2):
        [n_freqs, n_trials_1, n_chans, n_tsamps] = np.shape(train_data_1)
        n_trials_2 = np.size(train_data_2,1)

        self.reg_params_s = np.zeros(n_freqs)
        self.reg_params_t = np.zeros(n_freqs)

        # find the spatial and temporal means for each class 
        
        train_dists = np.zeros(((n_trials_1+n_trials_2), 4 * n_freqs))

        self.cov_means_s = np.zeros([n_freqs, 2, n_chans, n_chans])
        self.cov_means_t = np.zeros([n_freqs, 2, n_tsamps, n_tsamps])

        for il in range(n_freqs):
            # for each frequency band

            self.cov_means_s[il,:,:,:], self.reg_params_s[il] = utls.cal_riem_means(train_data_1[il,:,:,:], train_data_2[il,:,:,:], 'space', self.rType)
            self.cov_means_t[il,:,:,:], self.reg_params_t[il] = utls.cal_riem_means(train_data_1[il,:,:,:], train_data_2[il,:,:,:], 'time', self.rType)


            CovS_1 = utls.cal_covariance(train_data_1[il, :, :, :], 'space', self.reg_params_s[il])
            CovS_2 = utls.cal_covariance(train_data_2[il, :, :, :], 'space', self.reg_params_s[il])

            CovS = np.concatenate((CovS_1, CovS_2), axis=0)
            train_dists[:, il * 4] = np.mat(utls.cal_riemann_distance(CovS, self.cov_means_s[il, 0, :, :], self.rType)).transpose()
            train_dists[:, il * 4 + 1] = np.mat(utls.cal_riemann_distance(CovS, self.cov_means_s[il, 1, :, :], self.rType)).transpose()

            CovT_1 = utls.cal_covariance(train_data_1[il, :, :, :], 'time', self.reg_params_t[il])
            CovT_2 = utls.cal_covariance(train_data_2[il, :, :, :], 'time', self.reg_params_t[il])

            CovT = np.concatenate((CovT_1, CovT_2), axis=0)
            train_dists[:, il * 4 + 2] = np.mat(utls.cal_riemann_distance(CovT, self.cov_means_t[il, 0, :, :], self.rType)).transpose()
            train_dists[:, il * 4 + 3] = np.mat(utls.cal_riemann_distance(CovT, self.cov_means_t[il, 1, :, :], self.rType)).transpose()


        self.clf = sk.LogisticRegression(solver=self.logreg_solver)


        y1 = np.zeros(n_trials_1)
        y2 = np.ones(n_trials_2)
        y = np.concatenate((y1, y2))

        self.clf.fit(train_dists, y)

        train_P = self.clf.predict_proba(train_dists)

        return train_P

    def test(self,test_data_1, test_data_2):
        [n_freqs, n_trials_1, n_chans, n_tsamps] = np.shape(test_data_1)
        n_trials_2 = np.size(test_data_2,1)

        test_dists_1 = np.zeros((n_trials_1, 4 * n_freqs))
        test_dists_2 = np.zeros((n_trials_2, 4 * n_freqs))


        for il in range(n_freqs):
            # for each frequency band...

            CovS_1 = utls.cal_covariance(test_data_1[il, :, :, :], 'space', self.reg_params_s[il])
            CovS_2 = utls.cal_covariance(test_data_2[il, :, :, :], 'space', self.reg_params_s[il])

            test_dists_1[:, il * 4] = np.mat(utls.cal_riemann_distance(CovS_1, self.cov_means_s[il, 0, :, :], self.rType)).transpose()
            test_dists_1[:, il * 4 + 1] = np.mat(utls.cal_riemann_distance(CovS_1, self.cov_means_s[il, 1, :, :], self.rType)).transpose()
            test_dists_2[:, il * 4] = np.mat(utls.cal_riemann_distance(CovS_2, self.cov_means_s[il, 0, :, :], self.rType)).transpose()
            test_dists_2[:, il * 4 + 1] = np.mat(utls.cal_riemann_distance(CovS_2, self.cov_means_s[il, 1, :, :], self.rType)).transpose()


            CovT_1 = utls.cal_covariance(test_data_1[il, :, :, :], 'time', self.reg_params_t[il])
            CovT_2 = utls.cal_covariance(test_data_2[il, :, :, :], 'time', self.reg_params_t[il])

            test_dists_1[:, il * 4 + 2] = np.mat(utls.cal_riemann_distance(CovT_1, self.cov_means_t[il, 0, :, :], self.rType)).transpose()
            test_dists_1[:, il * 4 + 3] = np.mat(utls.cal_riemann_distance(CovT_1, self.cov_means_t[il, 1, :, :], self.rType)).transpose()
            test_dists_2[:, il * 4 + 2] = np.mat(utls.cal_riemann_distance(CovT_2, self.cov_means_t[il, 0, :, :], self.rType)).transpose()
            test_dists_2[:, il * 4 + 3] = np.mat(utls.cal_riemann_distance(CovT_2, self.cov_means_t[il, 1, :, :], self.rType)).transpose()


        test_P1 = self.clf.predict_proba(test_dists_1)
        test_P2 = self.clf.predict_proba(test_dists_2)

        return test_P1, test_P2

    def accuracy(self,test_data_1, test_data_2):
        # calculate the classification accuracy

        n_trials_1 = np.size(test_data_1, 1)
        n_trials_2 = np.size(test_data_1, 1)

        # find the scores for each classifier 
        test_DRMST_P1, test_DRMST_P2  =self.test(test_data_1, test_data_2)

        tempp = np.sum(test_DRMST_P1[:,0]>0.5) + np.sum(test_DRMST_P2[:,1]>0.5)

        # since classes are balanced, calculate the overall accuracy
        class_rate = tempp / float(n_trials_1 + n_trials_2)

        return class_rate

    def auc(self,test_data_1, test_data_2):
        # calculate the AUC

        n_trials_1 = np.size(test_data_1, 1)
        n_trials_2 = np.size(test_data_1, 1)

        # find the scores for each classifier 
        test_P1, test_P2  =self.test(test_data_1, test_data_2)
        y_scores = np.concatenate([np.asmatrix(test_P1[:,1]).transpose(), np.asmatrix(test_P2[:,1]).transpose()], axis=0)

        y1_true = np.zeros(n_trials_1)
        y2_true = np.ones(n_trials_2)
        y_true = np.concatenate((y1_true, y2_true))

        return  roc_auc_score(y_true, y_scores)

