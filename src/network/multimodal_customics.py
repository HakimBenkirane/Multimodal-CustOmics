# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the CustOMICS module.
"""
import numpy as np
import pandas as pd

from src.loss.survival_loss import CoxLoss
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from torch.optim import Adam

import shap
import tqdm

from src.datasets.multimodal_dataset import MultiModalDataset
from src.models.customics import CustOMICS
from src.models.graph_network import GNN_clusters
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.tasks.moe_network import MultiTaskMOE
from src.models.vae import VAE
from src.loss.classification_loss import classification_loss
from src.metrics.classification import (
    multi_classification_evaluation,
    plot_roc_multiclass,
)
from src.metrics.survival import CIndex_lifeline, cox_log_rank
from src.tools.utils import save_plot_score
from src.tools.utils import get_common_samples, get_sub_omics_df
from src.ex_vae.shap_vae import (
    processPhenotypeDataForSamples,
    randomTrainingSample,
    splitExprandSample,
    ModelWrapper,
    addToTensor,
)
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})


class MultimodalCustOMICS(nn.Module):
    def __init__(
        self,
        source_params,
        central_params,
        classif_params,
        surv_params,
        train_params,
        pathways,
        device,
        hallmarks_loc=None
    ):
        """
        Initializes a CustOMICS object.

        Args:
        - source_params (dict): Dictionary containing parameters for different data sources.
        - central_params (dict): Dictionary containing parameters for the central encoder/decoder.
        - classif_params (dict): Dictionary containing parameters for the classifier.
        - surv_params (dict): Dictionary containing parameters for survival prediction.
        - train_params (dict): Dictionary containing training parameters.
        - device (str): Device to be used for computation.

        Initializes the CustOMICS object with various components for multi-source data processing,
        including encoders, decoders, central encoders/decoders, classifiers, survival predictors,
        and training configurations.
        """
        super(MultimodalCustOMICS, self).__init__()
        self.n_source = len(list(source_params.keys()))
        self.pathways = pathways
        self.device = device
        self.graph_path = source_params['WSI']['WSI_path']
        self.wsi_clusters = source_params['WSI']['n_clusters']
        self.sources = [source for source in source_params.keys()]
        self.wsi_network = GNN_clusters(dim_features=source_params['WSI']['input_dim'],
                                             dim_target=source_params['WSI']['latent_dim'],
                                             layers=source_params['WSI']['hidden_dim'],
                                             pooling=source_params['WSI']['pooling'],
                                             dropout=source_params['WSI']['dropout'])
        self.multi_omics_networks = {pathway: CustOMICS(hallmark=pathway, 
                                                        source_params=source_params, 
                                                        central_params=central_params, 
                                                        classif_params=classif_params, 
                                                        surv_params=surv_params, 
                                                        train_params=train_params, 
                                                        device=self.device, hallmark_loc=hallmarks_loc) for pathway in self.pathways}
        self.rep_dim = source_params['WSI']['latent_dim'] + central_params['latent_dim']

        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim, 
                                                    hidden_dim=central_params['hidden_dim'], 
                                                    latent_dim=central_params['latent_dim'], 
                                                    norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'], 
                                                    hidden_dim=central_params['hidden_dim'], 
                                                    output_dim=self.rep_dim, 
                                                    norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self._set_central_layer()
        self.num_classes = classif_params["n_class"]
        self.lambda_classif = classif_params["lambda"]
        self.lambda_survival = surv_params["lambda"]
        self.downstream_net = MultiTaskMOE(pathways=self.pathways, spatial_clusters=list(range(self.wsi_clusters)), n_class=self.num_classes, latent_dim=central_params['latent_dim'])
        self.phase = 1
        self.switch_epoch = train_params["switch"]
        self.lr = train_params["lr"]
        self.optimizer = self._get_optimizer(self.lr)
        self.vae_history = []
        self.survival_history = []
        self.label_encoder = None
        self.one_hot_encoder = None

    def _get_optimizer(self, lr):
        """
        Creates and returns an Adam optimizer for the CustOMICS model.

        Args:
        - lr (float): Learning rate for optimization.

        Returns:
        - optimizer (torch.optim.Adam): Adam optimizer configured with parameters
        from autoencoders, central layer, survival predictor, and classifier.

        Constructs an Adam optimizer to manage the parameters involved in training the CustOMICS model.
        It combines parameters from various components, including autoencoders, central layers,
        survival predictors, and classifiers, using the specified learning rate for optimization.
        """
        lt_params = []
        for pathway, encoder in self.multi_omics_networks.items():
            for name, param in encoder.named_parameters():
                name = "{}_".format(pathway) + name.replace(".", " ")
                self.register_parameter(name=name, param=param)
        optimizer = Adam(self.parameters(), lr=lr)
        return optimizer
    
    def _set_central_layer(self):
        """
        Initializes the central variational autoencoder
        """
        self.central_layer = VAE(self.central_encoder,
                                                 self.central_decoder, self.device)
        

    def forward(self, graph, x):
        """
        Performs a forward pass through the CustOMICS model.

        Args:
        - x (list): List of input data for each data source.

        Returns:
        - dt_hat (dict): Dictionary containing reconstructed outputs for each data source.
        - dt_rep (dict): Dictionary containing latent representations for each data source.
        - mean (Tensor): Mean values from the central encoder.
        """
        pathways_rep = {pathway: encoder.forward_latent(x) for pathway, encoder in self.multi_omics_networks.items()}
        wsi_rep = self.wsi_network(graph)
        multimodal_rep = {pathway: {cluster: self.central_layer(
                                            torch.cat([pathway_rep, wsi_rep[:, cluster, :]], axis=1))[1] 
                                            for cluster in range(self.wsi_clusters)} for pathway, pathway_rep in pathways_rep.items()}
        return multimodal_rep

    def _compute_loss(self, graph, x):
        """
        Computes the loss function for the CustOMICS model based on the training phase.

        Args:
        - x (list): List of input data for each data source.

        Returns:
        - z (Tensor): Latent representation from the central encoder.
        - loss (Tensor): Total loss computed based on the phase of training.
        """
        loss = 0
        multimodal_rep = self.forward(graph, x)
        loss += sum([encoder.get_loss(x, self.phase) for encoder in self.multi_omics_networks.values()])
        return multimodal_rep, loss

    def _train_loop(self, graph, x, labels, os_time, os_event):
        """
        Runs the training loop for the CustOMICS model based on the input data and labels.

        Args:
        - x (list): List of input data for each data source.
        - labels (Tensor): Target labels for the input data.
        - os_time (Tensor): Observation time for survival analysis.
        - os_event (Tensor): Event indicator for survival analysis.

        Returns:
        - loss (Tensor): Computed loss value during the training loop.
        """
        self.optimizer.zero_grad()
        multimodal_rep, loss = self._compute_loss(graph, x)
        y_pred_prob, hazard_pred = self.downstream_net(multimodal_rep)
        survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
        classif_loss = classification_loss('CE', y_pred_prob, labels)
        loss += self.lambda_classif * classif_loss + self.lambda_survival * survival_loss
        return loss
   

    def fit(
        self,
        omics_train,
        clinical_df,
        label,
        event,
        surv_time,
        omics_val=None,
        batch_size=32,
        n_epochs=30,
        verbose=False,
    ):
        """
        Trains the CustOMICS model on the provided training data.

        Args:
        - omics_train (dict): Dictionary of training omics data for each source.
        - clinical_df (DataFrame): Clinical data associated with the omics data.
        - label (str): Column name specifying the label for the event.
        - event (str): Column name specifying the event indicator.
        - surv_time (str): Column name specifying the survival time.
        - omics_val (dict, optional): Dictionary of validation omics data. Defaults to None.
        - batch_size (int, optional): Batch size for training. Defaults to 32.
        - n_epochs (int, optional): Number of epochs for training. Defaults to 30.
        - verbose (bool, optional): Controls verbosity during training. Defaults to False.
        """

        encoded_clinical_df = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(
            encoded_clinical_df.loc[:, label].values
        )
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(
            encoded_clinical_df.loc[:, label].values
        )
        self.one_hot_encoder = OneHotEncoder(sparse=False).fit(
            encoded_clinical_df.loc[:, label].values.reshape(-1, 1)
        )

        kwargs = (
            {"num_workers": 2, "pin_memory": True} if self.device.type == "cuda" else {}
        )

        lt_samples_train = get_common_samples(
            [df for df in omics_train.values()] + [clinical_df]
        )
        dataset_train = MultiModalDataset(
            omics_df=omics_train,
            clinical_df=encoded_clinical_df,
            lt_samples=lt_samples_train,
            label=label,
            event=event,
            surv_time=surv_time,
            graph_path=self.graph_path
        )
        train_loader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=False, **kwargs
        )
        if omics_val:
            lt_samples_val = get_common_samples(
                [df for df in omics_val.values()] + [clinical_df]
            )
            dataset_val = MultiModalDataset(
                omics_df=omics_val,
                clinical_df=encoded_clinical_df,
                lt_samples=lt_samples_val,
                label=label,
                event=event,
                surv_time=surv_time,
                graph_path=self.graph_path
            )
            val_loader = DataLoader(
                dataset_val, batch_size=batch_size, shuffle=False, **kwargs
            )

        self.history = []
        for epoch in range(n_epochs):
            overall_loss = 0
            for batch_idx, (graph, x, labels, os_time, os_event) in tqdm.tqdm(enumerate(train_loader)):
                self.train()
                loss_train = self._train_loop(graph, x, labels, os_time, os_event)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx + 1) * batch_size)
            overall_loss = 0
            if val_loader != None:
                for batch_idx, (graph, x, labels, os_time, os_event) in enumerate(val_loader):
                    self.eval()
                    loss_val = self._train_loop(graph, x, labels, os_time, os_event)
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / ((batch_idx + 1) * batch_size)

                self.history.append((average_loss_train, average_loss_val))
                if verbose:
                    print(
                        "\tEpoch",
                        epoch + 1,
                        "complete!",
                        "\tAverage Loss Train : ",
                        average_loss_train,
                        "\tAverage Loss Val : ",
                        average_loss_val,
                    )
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print(
                        "\tEpoch",
                        epoch + 1,
                        "complete!",
                        "\tAverage Loss Train : ",
                        average_loss_train,
                    )

    def get_latent_representation(self, omics_df, tensor=False):
        """
        Retrieves latent representations from the CustOMICS model for the provided omics data.

        Args:
        - omics_df (dict): Dictionary of omics data for each source.
        - tensor (bool, optional): Flag to return data as Tensor. Defaults to False.

        Returns:
        - z (ndarray): Latent representations obtained from the CustOMICS model.
        """
        self.eval_all()
        if tensor == False:
            x = [torch.Tensor(omics_df[source].values) for source in omics_df.keys()]
        else:
            x = [omics for omics in omics_df]
        with torch.no_grad():
            for i in range(len(x)):
                x[i] = x[i].to(self.device)
            z, loss = self._compute_loss(x)
        return z.cpu().detach().numpy()

    def plot_representation(
        self, omics_df, clinical_df, labels, filename, title, show=True
    ):
        """
        Plots the latent representation obtained from the CustOMICS model.

        Args:
        - omics_df (dict): Dictionary of omics data for each source.
        - clinical_df (DataFrame): Clinical data associated with the omics data.
        - labels (str): Column name specifying the labels for visualization.
        - filename (str): File name to save the plot.
        - title (str): Title for the plot.
        - show (bool, optional): Flag to display the plot. Defaults to True.
        """
        labels_df = clinical_df.loc[:, labels]
        lt_samples = get_common_samples(
            [df for df in omics_df.values()] + [clinical_df]
        )
        omics_df = get_sub_omics_df(omics_df, lt_samples)
        z = self.get_latent_representation(omics_df=omics_df)
        save_plot_score(filename, z, labels_df[lt_samples].values, title, show=True)

    def source_predict(self, expr_df, source):
        """
        Predicts outcomes using the CustOMICS model for a specific data source.

        Args:
        - expr_df (DataFrame): Input expression data.
        - source (str): Data source identifier.

        Returns:
        - y_pred_proba (Tensor): Predicted probabilities of outcomes.
        """
        z = self.encoders[source](expr_df)
        y_pred_proba = self.classifier(z)
        return y_pred_proba



    def evaluate(
        self,
        omics_test,
        clinical_df,
        label,
        event,
        surv_time,
        task,
        batch_size=32,
        plot_roc=False,
    ):
        """
        Evaluates the CustOMICS model.

        Args:
        - omics_test (dict): Dictionary of test omics data for each source.
        - clinical_df (DataFrame): Clinical data associated with the omics data.
        - label (str): Column name specifying the label for evaluation.
        - event (str): Column name specifying the event indicator.
        - surv_time (str): Column name specifying the survival time.
        - task (str): Task type for evaluation ('survival' or 'classification').
        - batch_size (int, optional): Batch size for evaluation. Defaults to 32.
        - plot_roc (bool, optional): Flag to plot ROC curves. Defaults to False.

        Returns:
        - evaluation_metrics (list or float): Evaluation metrics based on the chosen task.
        """

        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(
            encoded_clinical_df.loc[:, label].values
        )

        kwargs = (
            {"num_workers": 2, "pin_memory": True} if self.device.type == "cuda" else {}
        )

        lt_samples_test = get_common_samples(
            [df for df in omics_test.values()] + [clinical_df]
        )
        dataset_test = MultiModalDataset(
                omics_df=omics_test,
                clinical_df=encoded_clinical_df,
                lt_samples=lt_samples_test,
                label=label,
                event=event,
                surv_time=surv_time,
                graph_path=self.graph_path
            )
        test_loader = DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, **kwargs
        )

        self.eval()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (graph, x, labels, os_time, os_event) in enumerate(test_loader):
                z, loss = self._compute_loss(graph, x)
                if task == "survival":
                    predicted_survival_hazard = self.downstream_net(z)[1]
                    predicted_survival_hazard = (
                        predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    )
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(
                        CIndex_lifeline(predicted_survival_hazard, os_event, os_time)
                    )
                    return np.mean(c_index)
                elif task == "classification":
                    y_pred_proba = self.downstream_net(z)[0]
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    print(y_pred_proba)
                    y_true = labels.cpu().detach().numpy()
                    classif_metrics.append(
                        multi_classification_evaluation(
                            y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder
                        )
                    )
                    if plot_roc:
                        plot_roc_multiclass(
                            y_test=y_true,
                            y_pred_proba=y_pred_proba,
                            filename="test",
                            n_classes=self.num_classes,
                            var_names=np.unique(
                                clinical_df.loc[:, label].values.tolist()
                            ),
                        )

                    return classif_metrics

    def stratify(
        self,
        omics_df,
        clinical_df,
        event,
        surv_time,
        treshold=0.5,
        save_plot=False,
        plot_title="",
        filename="",
    ):
        """
        Stratifies samples based on predicted risk, generates Kaplan-Meier plots, and calculates p-value.

        Args:
        - omics_df (dict): Dictionary of omics data for each source.
        - clinical_df (DataFrame): Clinical data associated with the omics data.
        - event (str): Column name specifying the event indicator.
        - surv_time (str): Column name specifying the survival time.
        - threshold (float, optional): Threshold for risk stratification. Defaults to 0.5.
        - save_plot (bool, optional): Flag to save the Kaplan-Meier plot. Defaults to False.
        - plot_title (str, optional): Title for the plot. Defaults to "".
        - filename (str, optional): Name for the saved plot file. Defaults to "".
        """

        lt_samples = get_common_samples(
            [df for df in omics_df.values()] + [clinical_df]
        )
        z = self.get_latent_representation(omics_df)
        hazard_pred = self.survival_predictor(torch.Tensor(z)).cpu().detach().numpy()
        dt_strat = {"high": [], "low": []}
        clinical_df["risk"] = ""
        for i in range(len(lt_samples)):
            if hazard_pred[i] <= np.mean(hazard_pred):
                clinical_df.loc[lt_samples[i], "risk"] = "low"
                dt_strat["low"].append(lt_samples[i])
            else:
                clinical_df.loc[lt_samples[i], "risk"] = "high"
                dt_strat["high"].append(lt_samples[i])
        print(clinical_df["risk"])
        clinical_df.to_csv("clinical_risk.csv")
        plt.close("all")
        kmf_low = KaplanMeierFitter(label="low risk")
        kmf_high = KaplanMeierFitter(label="high risk")
        kmf_low.fit(
            clinical_df.loc[dt_strat["low"], surv_time],
            clinical_df.loc[dt_strat["low"], event],
        )
        kmf_high.fit(
            clinical_df.loc[dt_strat["high"], surv_time],
            clinical_df.loc[dt_strat["high"], event],
        )
        p_value = cox_log_rank(
            hazard_pred.reshape(1, -1)[0],
            np.array(clinical_df.loc[lt_samples, event].values, dtype=float),
            np.array(clinical_df.loc[lt_samples, surv_time].values, dtype=float),
        )

        kmf_low.plot()
        kmf_high.plot()
        plt.title(plot_title + " (p-value = {:.3g})".format(p_value))
        plt.xlim((0, 2500))
        if save_plot:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    def explain(
        self,
        sample_id,
        omics_df,
        clinical_df,
        source,
        subtype,
        label="PAM50",
        device="cpu",
        show=False,
        get_names=False,
        sample=False,
    ):
        """
        Generates SHAP summary plots to explain model predictions.

        Args:
        - sample_id (list): List of sample IDs.
        - omics_df (dict): Dictionary of omics data for each source.
        - clinical_df (DataFrame): Clinical data associated with the omics data.
        - source (str): Data source identifier.
        - subtype (str): Subtype for analysis.
        - label (str, optional): Column name specifying the label for analysis. Defaults to 'PAM50'.
        - device (str, optional): Device for computation. Defaults to 'cpu'.
        - show (bool, optional): Flag to display the plot. Defaults to False.
        - get_names (bool, optional): Flag to retrieve feature names. Defaults to False.
        - sample (bool, optional): Flag to generate a single-sample SHAP plot. Defaults to False.
        """
        expr_df = omics_df[source]
        sample_id = list(set(sample_id).intersection(set(expr_df.index)))
        phenotype = processPhenotypeDataForSamples(
            clinical_df, sample_id, self.label_encoder
        )
        conditionaltumour = phenotype.loc[:, label] == subtype

        expr_df = expr_df.loc[sample_id, :]
        normal_expr = randomTrainingSample(expr_df, 100)
        tumour_expr = splitExprandSample(
            condition=conditionaltumour, sampleSize=100, expr=expr_df
        )
        # put on device as correct datatype
        background = addToTensor(expr_selection=normal_expr, device=device)
        male_expr_tensor = addToTensor(expr_selection=tumour_expr, device=device)

        e = shap.DeepExplainer(ModelWrapper(self, source=source), background)
        shap_values_female = e.shap_values(male_expr_tensor, ranked_outputs=None)
        if not sample:
            shap.summary_plot(
                shap_values_female[0],
                features=tumour_expr,
                feature_names=list(tumour_expr.columns),
                show=False,
                plot_type="violin",
                max_display=10,
                plot_size=[4, 6],
            )
            plt.savefig("shap_{}_{}.png".format(source, subtype), bbox_inches="tight")
        if show:
            plt.show()
        if get_names:
            feature_names = expr_df.columns
            print(feature_names)
            rf_resultX = pd.DataFrame(shap_values_female[0], columns=feature_names)
            vals = np.abs(rf_resultX.values).mean(0)
            shap_importance = pd.DataFrame(
                list(zip(feature_names, vals)),
                columns=["col_name", "feature_importance_vals"],
            )
            shap_importance.sort_values(
                by=["feature_importance_vals"], ascending=False, inplace=True
            )
            print(shap_importance)
            shap_importance.to_csv("shap_{}_{}_survival.csv".format(source, subtype))
        if sample:
            shap.plots._waterfall.waterfall_legacy(
                e.expected_value[0],
                shap_values_female[0][0],
                feature_names=expr_df.columns.tolist(),
                show=False,
            )
            plt.savefig("shap_{}_{}.png".format(source, subtype), bbox_inches="tight")
        plt.clf()

    def plot_loss(self):
        """
        Plots the evolution of the loss function with respect to epochs.
        """
        n_epochs = len(self.history)
        plt.title("Evolution of the loss function with respect to the epochs")
        plt.vlines(
            x=self.switch_epoch,
            ymin=0,
            ymax=2.5,
            colors="purple",
            ls="--",
            lw=2,
            label="phase 2 switch",
        )
        plt.plot(
            range(0, n_epochs), [loss[0] for loss in self.history], label="train loss"
        )
        plt.plot(
            range(0, n_epochs), [loss[1] for loss in self.history], label="val loss"
        )
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()




