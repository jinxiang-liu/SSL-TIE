import numpy as np
class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01, return_masks=False):
        # gtmap: 所有bbox里面的值，都设定为1，其他的设定为0
        # infer_map = np.zeros((112, 112)) 
        infer_map = np.zeros((224, 224)) 
        infer_map[infer>=thres] = 1  # 预测：经过thresholding之后，预测的positive的点设置为1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))   # 
        # IoU: intersaction：通过两个二值图相乘即可， Union: GTmap的像素数目+ (pred的像素数目扣除二者重叠部分的像素数目 )

        self.ciou.append(ciou)

        # Return three masks for vis: intersection mask, gt excluding intersection, prediction excluding intersection
        if return_masks:
            inter_mask = infer_map*gtmap
            gt_exclude_inter = gtmap - inter_mask
            pred_exclude_inter = infer_map - inter_mask

            return ciou, np.sum(infer_map*gtmap),(np.sum(gtmap)+np.sum(infer_map*(gtmap==0))), \
                ( inter_mask.astype(bool), gt_exclude_inter.astype(bool), pred_exclude_inter.astype(bool), infer_map )

        else:
            return ciou, np.sum(infer_map*gtmap),(np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))


    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou)>=0.05*i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = sklearn.metrics.auc(x, results)
        print(results)
        return auc

    def final(self):
        ciou = np.mean(np.array(self.ciou)>=0.5)
        return ciou

    def clear(self):
        self.ciou = []