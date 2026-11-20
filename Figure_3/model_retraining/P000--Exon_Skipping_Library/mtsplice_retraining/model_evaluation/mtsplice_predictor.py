from mtsplice_retraining.model_evaluation import scaling
import numpy as np
from mtsplice_retraining.model_evaluation.r2_calc import (
    calc_r2_with_nan_tsplice as calc_r2
)
from mtsplice_retraining.model_evaluation.print_r2 import (
    print_table
)
from src.esl import cell_types
from collections import defaultdict


# Could potentially inherit from this
class TupleDict():
    def __init__(self, order, update=True):
        self.pr = {}
        self.order = order
        self.order_dict = {val : i for i, val in enumerate(order)}
        self.update = update
        return

    def __getitem__(self, key):
        return self.pr[key]

    def __contains__(self, key):
        return key in self.pr

    def __iter__(self):
        return iter(self.pr)

    def add(self, value, **kwargs):
        key = tuple(kwargs.get(k, "") for k in self.order)
        if not self.update and key in self.pr:
            raise KeyError(f"Key already exists: {key}")
        self.pr[key] = value
        return
        
    def get(self, **kwargs):
        key = tuple(kwargs.get(k, "") for k in self.order)
        return self.pr[key]

    def filter(self, **kwargs):
        """ 
        Returns predictions that satisfy
        specific values for any number of fields.
        """
        filtered = {}
        for key in self.pr:
            for arg in kwargs:
                index = self.order_dict[arg]
                value = kwargs[arg]
                if key[index] == value:
                    filtered = self.pr[key]
        return filtered


class Predictions():
    """
    Helper class to keep track of 
    predictions and use keyword 
    arguments to extract them, rather
    than relying on exact ordered dictionary keys.
    """

    def __init__(self,):
        self.pr = {}
        self.order = [
            "scale",
            "model",
            "head",
            "data", 
            "version_t",
            "version_mm",
            "version_mt",
            "output",
        ]
        return

    def add_prediction(self, prediction, **kwargs):
        key = tuple(kwargs.get(k, "") for k in self.order)
        if key not in self.pr:
            self.pr[key] = prediction
        return

    def add_predictions(self, predictions):
        for key in predictions:
            if key not in self.pr:
                self.pr[key] = predictions[key]
        return

    def check_key(self, key):
        if key in self.pr:
            return
        raise KeyError(f"{key}, {self.order}")
        
    def get_prediction(self, **kwargs):
        key = tuple(kwargs.get(k, "") for k in self.order)
        self.check_key(key)
        return self.pr[key]

    def __contains__(self, key):
        return key in self.pr
    
    def __getitem__(self, key):
        self.check_key(key)
        return self.pr[key]

    def __iter__(self):
        return iter(self.pr)

    def get_true_data(
        self,
        data="",
        scale="",
        model="",
        **kwargs,
        ):
        key = (
            scale,
            model,
            "",
            data,
            "",
            "",
            "",
            "true",
        )
        self.check_key(key)
        return self.pr[key]

    def get_scaling_factor(
        self,
        data="",
        model="",
        ):
        key = (
            "",
            model,
            "",
            data,
            "",
            "",
            "",
            "scaling",
        )
        self.check_key(key)
        return self.pr[key]
        

class MTSplicePredictor():
    """
    Helper class to make predictions and scale them
    """

    def __init__(
        self,
        tsplice_data,
        mtsplice_data,
        tsplice_base_scale="dlogit",
        mtsplice_base_scale="logit",
        mmsplice_retrained="",
        tsplice_output_scales=None,
        mtsplice_output_scales=None,
        hyperband=True,
        train_combined=False,
        debug=False,
        **kwargs,
        ):
        self.tsplice_data = tsplice_data
        self.mtsplice_data = mtsplice_data
        self.data = {
            "tsplice" : self.tsplice_data,
            "mtsplice" : self.mtsplice_data,
        }
        self.tsplice_base_scale = tsplice_base_scale
        self.mtsplice_base_scale = mtsplice_base_scale
        self.mmsplice_retrained = mmsplice_retrained
        self.tsplice_output_scales = tsplice_output_scales
        self.mtsplice_output_scales = mtsplice_output_scales
        self.output_scales = {
            "tsplice" : tsplice_output_scales,
            "mtsplice" : mtsplice_output_scales
        }

        self.model_info = {
            "tsplice" : {
                    "base_scale" : tsplice_base_scale,
                    "output_scales" : tsplice_output_scales,
                    "data" : self.tsplice_data,
                    "hyperband" : hyperband,
                    "versions" : {},
                    "version_keys" : {},
                },
            "mtsplice" : {
                    "base_scale" : mtsplice_base_scale,
                    "output_scales" : mtsplice_output_scales,
                    "data" : self.mtsplice_data,
                    "hyperband" : False,
                    "versions" : {},
                    "version_keys" : {},
                }
        }
        self.hyperband = hyperband
        self.train_combined = train_combined
        self.debug = debug
        self.predictions = Predictions()
        self.datasets = ["train", "val", "test"]
        self.version_keys = ["tsplice", "mmsplice", "mtsplice", "head"]
        self.order_dict = {
            val : i 
            for (i, val) in enumerate(self.predictions.order)
        }
        self.init_scaling_factor()
        self.init_true_data()
        self.init_sequence_input()
        return

    def init_true_data(self):
        """
        Puts the y data into the Prediction expected format.
        """
        tsplice_data = self.tsplice_data
        mtsplice_data = self.mtsplice_data
        scaled_aggregated_t = defaultdict(list)
        scaled_aggregated_mt = defaultdict(list)
        for dataset in self.datasets:
            true_t = tsplice_data["y_" + dataset]
            scaled_true_t = self.scale_(
                data=true_t,
                model_name="tsplice",
                dataset=dataset
            )
            for scale in scaled_true_t:
                self.predictions.add_prediction(
                    scaled_true_t[scale],
                    scale=scale,
                    model="tsplice",
                    head="",
                    data=dataset,
                    version_t="",
                    version_mm="",
                    version_mt="",
                    output="true",
                )
                scaled_aggregated_t[scale].append(scaled_true_t[scale])

            true_mt = mtsplice_data["y_" + dataset]
            scaled_true_mt = self.scale_(
                data=true_mt,
                model_name="mtsplice",
                dataset=dataset
            )
            for scale in scaled_true_mt:
                self.predictions.add_prediction(
                    scaled_true_mt[scale],
                    scale=scale,
                    model="mtsplice",
                    head="",
                    data=dataset,
                    version_t="",
                    version_mm="",
                    version_mt="",
                    output="true",
                )
                scaled_aggregated_mt[scale].append(scaled_true_mt[scale])
        for scale in scaled_aggregated_t:
                agg = np.concatenate(scaled_aggregated_t[scale])
                self.predictions.add_prediction(
                    agg,
                    scale=scale,
                    model="tsplice",
                    head="",
                    data="full",
                    version_t="",
                    version_mm="",
                    version_mt="",
                    output="true",
                )
        for scale in scaled_aggregated_mt:
                agg = np.concatenate(scaled_aggregated_mt[scale])
                self.predictions.add_prediction(
                    agg,
                    scale=scale,
                    model="mtsplice",
                    head="",
                    data="full",
                    version_t="",
                    version_mm="",
                    version_mt="",
                    output="true",
                )
        return

    def init_sequence_input(self):
        """
        Remove cell mean from TSplice inputs.
        """
        for dataset in self.datasets:
            data = self.data["tsplice"]["x_" + dataset]
            self.data["tsplice"]["x_" + dataset] = [
                data[0],
                data[1]
            ]
            data = self.data["mtsplice"]["x_" + dataset]
            self.data["mtsplice"]["x_" + dataset] = [
                *data[0:10],
                *data[12:14],
                *data[14:15],
            ]
        return

    def init_scaling_factor(self):
        """
        Puts the scaling data into the Prediction expected format.
        e.g. need ref values to go from delta values to raw values
        """
        tsplice_data = self.tsplice_data
        mtsplice_data = self.mtsplice_data
        cell_logits = []
        ref_cell_psis = []
        for dataset in self.datasets:
            cell_mean_logit = tsplice_data["x_" + dataset][-1]
            cell_logits.append(cell_mean_logit)
            self.predictions.add_prediction(
                cell_mean_logit,
                scale="",
                model="tsplice",
                head="",
                data=dataset,
                version_t="",
                version_mm="",
                version_mt="",
                output="scaling",
            )

            ref_cell_psi = mtsplice_data["psi_" + dataset]
            ref_cell_psis.append(ref_cell_psi)
            self.predictions.add_prediction(
                ref_cell_psi,
                scale="",
                model="mtsplice",
                head="",
                data=dataset,
                version_t="",
                version_mm="",
                version_mt="",
                output="scaling",
            )
        cell_agg = np.concatenate(cell_logits)
        ref_psi_agg = np.concatenate(ref_cell_psis)
        self.predictions.add_prediction(
            cell_agg,
            scale="",
            model="tsplice",
            head="",
            data="full",
            version_t="",
            version_mm="",
            version_mt="",
            output="scaling",
        )
        self.predictions.add_prediction(
            ref_psi_agg,
            scale="",
            model="mtsplice",
            head="",
            data="full",
            version_t="",
            version_mm="",
            version_mt="",
            output="scaling",
        )
        return

    def add_model(
        self,
        model,
        model_name,
        version_t,
        version_mm,
        version_mt,
        head="",
        ):
        """
        Add model to the set of possible
        models that make predictions.
        version_t: 'baseline' if not retrained, 'retrained'
        version_mm: 'baseline' if not retrained, 'retrained'.
            The retrained version is made by loading weights.
        version_mt: If the model was trained while combined,
            (i.e. not MMSplice and TSplice separately), then
            this will be 'retrained'. Otherwise, it will be empty.
        head: empty if MTSplice model, but could be the model head
            ('0' to '4') or 'ensemble'.
        """
        if model_name == "tsplice" and not head:
            raise ValueError("define model head for tsplice") 
        version_tuple = (version_t, version_mm, version_mt, head)
        if version_tuple in self.model_info[model_name]:
            raise KeyError("model version {version_tuple} exists")
        version_dict = {
            "model_name" : model_name,
            "version" : {
                "tsplice" : version_t, 
                "mmsplice" : version_mm,
                "mtsplice" : version_mt,
                "head" : head,
            },
        }
        self.model_info\
            [model_name]\
            ["versions"]\
            [version_tuple] = model
        self.model_info\
            [model_name]\
            ["version_keys"]\
            [version_tuple] = version_dict
        return

    def replace_key(self, key, field, new_value):
        """
        Replace a field in a tuple-based key.
        Tuples are immutable (and thus hashable)
        so need to make a copy.
        """
        index = self.order_dict[field]
        left = key[:index]
        right = key[index + 1:]
        return tuple(left + (new_value,) + right)

    def scale_(
        self,
        data,
        model_name,
        scaling_factor=None,
        dataset=None,
        ):
        if scaling_factor is None:
            assert dataset is not None
            scaling_factor = (
                self
                .predictions
                .get_scaling_factor(
                    model=model_name,
                    data=dataset
                    )
            )
        scaled = {}
        if model_name == "tsplice":
            scaled["dlogit"] = data
            scaled["psi"] = scaling.tsplice_psi_scale(
                data,
                scaling_factor
            )
            scaled["logit"] = scaling.tsplice_natural_scale(
                data,
                scaling_factor
            )
            scaled["dpsi"] = scaling.tsplice_dpsi_scale(
                data,
                scaling_factor
            )
        elif model_name == "mtsplice":
            scaled["logit"] = data
            scaled["psi"] = scaling.mtsplice_psi_scale(
                data,
                scaling_factor
            )
            scaled["dlogit"] = scaling.mtsplice_dlogit_scale(
                data,
                scaling_factor
            )
            scaled["dpsi"] = scaling.mtsplice_dpsi_scale(
                data,
                scaling_factor
            )
        return scaled
        

    def scale_data(self, curr_pr, model, dataset):
        """
        Scale predictions into dlogit, logit, psi and dpsi scale.
        
        Currently assumes TSplice is trained with data,
        dlogit, and MTSplice is trained on (variant) logit
        data. This likely won't be changed unless necessary
        to train TSplice on other data. Namely, MTSplice scaling
        is only defined on the logit scale in the original paper.
        """
        model_name = model["model_name"]
        scaling_factor = (
            self
            .predictions
            .get_scaling_factor(
                model=model_name,
                data=dataset
                )
        )
        predictions = self.scale_(
            data=curr_pr,
            model_name=model_name,
            scaling_factor=scaling_factor
        )
        scales = predictions.keys()
        scaled_predictions = {}
        for scale in scales:
            key_pr = self.make_prediction_key(
                model=model,
                dataset=dataset,
                scale=scale,
                output="prediction",
            )
            key_r2 = self.replace_key(
                key_pr,
                field="output",
                new_value="r2"
            )
            try:
                y = self.predictions.get_true_data(
                    data=dataset,
                    scale=scale,
                    model=model_name,
                )
            except KeyError:
                breakpoint()
            r2 = calc_r2(predictions[scale], y)
            scaled_predictions[key_pr] = predictions[scale]
            scaled_predictions[key_r2] = r2
        return scaled_predictions 

    def reference_working_model(self, model):
        """
        Returns the correct model given the
        'model' query, which is a dictionary
        containing the model name and version.
        """
        model_name = model["model_name"]
        model_version_tuple = tuple(
            model["version"][k] for k in self.version_keys
        )
        return self.model_info[model_name]["versions"][model_version_tuple]

    def get_scales(self, model):
        """
        Check which output scales to 
        convert predictions to.
        Currently just converting everything,
        may or may not change this. The base scale
        is currently under self.model_info
        """
        model_name = model["model_name"]
        return self.model_info[model_name]["output_scales"]

    def add_tsplice_all_heads(
        self,
        tsplice_models,
        version_t="",
        version_mm="",
        version_mt="",
        ):
        """
        Specifically add models but for tsplice.
        """
        for i, model in enumerate(tsplice_models):
            if i > 4 :
                head = "ensemble"
            else:
                head = str(i)
            self.add_model(
                model=model,
                model_name="tsplice",
                version_t=version_t,
                version_mm=version_mm,
                version_mt=version_mt,
                head=head
            )
        return

    def predict_data(self, models=None):
        """
        Main function to predict data for all
        models in query models or in self.model_info.
        Makes the predictions, which are in the base_scale
        and then produces scaled predictions for all available
        scales. Adds predictions to a Prediction object. 
        """
        if models is None:
            tsplice_versions = self.model_info["tsplice"]["version_keys"]
            mtsplice_versions = self.model_info["mtsplice"]["version_keys"]
            tsplice_models = [
                tsplice_versions[version]
                for version in tsplice_versions
            ]
            mtsplice_models = [
                mtsplice_versions[version]
                for version in mtsplice_versions
            ]
            models = tsplice_models + mtsplice_models
        for model in models:
            working_model = self.reference_working_model(model) 
            scaled_predictions = None
            for dataset in self.datasets:
                key = self.make_prediction_key(
                    model=model,
                    dataset=dataset,
                    output="prediction",
                )
                if key in self.predictions:
                    continue
                model_name = model["model_name"]
                if self.debug:
                    print("DEBUG mode dummy prediction")
                    unscaled_pr = self.dummy_predict(
                        self.data[model_name]["x_" + dataset]
                    )
                else:
                    unscaled_pr = working_model.predict(
                        self.data[model_name]["x_" + dataset]
                    )
                scaled_predictions = self.scale_data(
                    unscaled_pr,
                    model,
                    dataset,
                )
                self.predictions.add_predictions(scaled_predictions)
            # will be None if all predictions already exist
            if scaled_predictions is not None:
                full_scaled, r2 = self.aggregate_data(
                    scaled_predictions,
                    model_name,
                ) 
                self.predictions.add_predictions(full_scaled)
                self.predictions.add_predictions(r2)
        return

    def aggregate_data(self, scaled_predictions, model_name):
        """
        combines predictions for 'train', 'val', 'test' into 'full'
        in the order train, val, test. Repeat for each scale and
        make the appropriate prediction key.
        """
        scales = [key[self.order_dict["scale"]] for key in scaled_predictions]
        for key in scaled_predictions:
            ex_key = key
            break
        full_scaled = {}
        r2 = {}
        for scale in scales:
            full = []
            for dataset in self.datasets:
                key = self.replace_key(ex_key, "scale", scale)
                key = self.replace_key(key, "data", dataset)
                d = self.predictions[key]
                full.append(d)
            key = self.replace_key(key, "data", "full")
            full_scaled[key] = np.concatenate(full)
            true_agg = self.predictions.get_true_data(
                data="full",
                scale=scale,
                model=model_name,
            )
            r2_key = self.replace_key(key, "output", "r2")
            try:
                r2[r2_key] = calc_r2(full_scaled[key], true_agg)
            except IndexError:
                breakpoint()
                " check sizes "

        return full_scaled, r2

    def dummy_predict(self, data):
        shape = (data[0].shape[0], 5)
        return np.random.random(size=shape)

    def make_prediction_key(
        self,
        model,
        dataset,
        scale="",
        output="prediction",
        ):
        """
        Makes prediction key using information
        from the 'model' query, which is a dictionary
        containing the model name and version.
        Note the order of the keys in the tuple must
        match the Prediction expected structure.
        """
        model_name = model["model_name"]
        if not scale:
            scale = self.model_info[model_name]["base_scale"]
        head = model["version"]["head"]
        version_t = model["version"]["tsplice"]
        version_mm = model["version"]["mmsplice"]
        version_mt = model["version"]["mtsplice"]
        data = dataset
        return (
            scale,
            model_name,
            head,
            data,
            version_t,
            version_mm,
            version_mt,
            output,
        )

    def get_prediction(self, **kwargs):
        """ Pass key info to Prediction object. """
        if kwargs["model"] == "tsplice":
            assert kwargs["head"]
        return self.predictions.get_prediction(**kwargs)

    def filter_predictions(self, **kwargs):
        """ 
        Returns predictions that satisfy
        specific values for any number of fields.
        """
        filtered = {}
        for key in self.predictions:
            all_match = True
            for arg in kwargs:
                index = self.order_dict[arg]
                value = kwargs[arg]
                if key[index] != value:
                    all_match = False
            if all_match:
                filtered[key] = self.predictions[key]
        return filtered

    def make_tsplice_table(
        self,
        include_ensemble=True,
        version_t="baseline",
        version_mm="baseline",
        version_mt="",
        scale="dlogit"
        ):
        """
        Pulls predictions for TSplice model to 
        make a data table showing R^2 values
        for the specified model.
        """
        out_ = []
        heads = [str(i) for i in range(4)]
        if include_ensemble:
            heads.append("ensemble")
        for h in heads:
            r2_data = {}
            for dataset in self.datasets:
                r2_data[dataset] = self.get_prediction(
                    scale=scale,
                    data=dataset,
                    model="tsplice",
                    version_t=version_t,
                    version_mm=version_mm,
                    version_mt=version_mt,
                    head=h,
                    output="r2"
               )
            scale_title = scale.replace("d", "Delta ")
            scale_title = scale_title.title()
            scale_title.replace("psi", "PSI")
            version_title = version_t.title()
            if version_mt:
                version_title += " Final"
            table_title = (
                f"{version_title} "
                f"TSplice {h.title()} "
                f"R^2 {scale_title}"
            )
            table = "\n".join(print_table(
                cell_types,
                r2_data,
                title=table_title,
                print_string=False,
            ))
            out_.append(table)
        return "\n\n".join(out_)


    def make_mtsplice_table(
            self,
            version_t="baseline",
            version_mm="baseline",
            version_mt="",
            scale="dlogit"
        ):
        """
        Pulls predictions for TSplice model to 
        make a data table showing R^2 values
        for the specified model.
        """
        out_ = []
        r2_data = {}
        for dataset in self.datasets:
            r2_data[dataset] = self.get_prediction(
                scale=scale,
                data=dataset,
                model="mtsplice",
                version_t=version_t,
                version_mm=version_mm,
                version_mt=version_mt,
                head="",
                output="r2"
           )
        scale_title = scale.replace("d", "Delta ")
        scale_title = scale_title.title()
        scale_title.replace("psi", "PSI")
        version_title = version_t.title()
        if version_mt:
            version_title += " Final"
        table_title = (
            f"{version_title} "
            f"MTSplice "
            f"R^2 {scale_title}"
        )
        table = "\n".join(print_table(
            cell_types,
            r2_data,
            title=table_title,
            print_string=False,
        ))
        out_.append(table)
        return "\n\n".join(out_)

    def save_predictions(self, path):
        return

    def get_scale_title(self, scale):
        scale_title = scale.replace("d", "Δ")
        scale_title = scale_title.replace("logit", "Logit")
        scale_title = scale_title.replace("psi", "PSI")
        return scale_title

    def make_plot_title(
            self,
            model="",
            scale="",
            head="",
            version_t="",
            version_mm="",
            version_mt="",
            **kwargs,
        ):
        title = []
        scale_title = self.get_scale_title(scale)
        scale = f"{scale_title} Model Predictions for\n"
        if model == "tsplice":
            title = f"TSplice {head.title()} {version_t.title()}"
            if version_mt:
                title += " (Retrained with MMSplice)"
            return scale + title
        elif model == "mtsplice":
            if version_t:
                title.append(f"TSplice {version_t.title()}")
            if version_mm and model == "mtsplice":
                title.append(f"MMSplice {version_mm.title()}")
            if version_mt:
                title.append("Retrained Combined Model")

        title = ", ".join(title)
        return scale + title

    def make_subplot_title(
            self,
            scale="",
            data="",
            cell="",
            **kwargs,
        ):
        scale_title = self.get_scale_title(scale)
        subplot_title = (
            f"{data.title()} "
            f"{cell} "
            f"{scale_title}"
        )
        return f"{data.title()} " 

    def make_axis_label(
        self,
        label,
        model="",
        scale="",
        **kwargs,
    ):
        model_ = model.split("splice")
        model = model_[0].upper() + model[len(model_) - 1:].title()
        scale = self.get_scale_title(scale)
        return f"{model} {label} {scale}"

    def make_plot_save_name(
        self,
        datasets="",
        model="",
        scale="",
        head="",
        version_t="",
        version_mm="",
        version_mt="",
        **kwargs,
    ):
        if head:
            model_save = model + "_" + head
        else:
            model_save = model
        if model == "tsplice":
            version = version_t
        elif model == "mtsplice":
            version = version_mm

        if version_mt:
            version += "_combined"

        datasets_save = "_".join(datasets)
        
        return (
            f"{model_save}_{version}_{scale}_prediction_"
            f"observation_scatter_{datasets_save}"
        )

    def make_barplot():
        return



