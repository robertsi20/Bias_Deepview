from DeepView.deepview.DeepView import DeepViewSelector

from DeepView.deepview.Selector import SelectFromCollection

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
import warnings


class DeepViewBias(DeepViewSelector):

    def __init__(self, *args, class_dict=None, sensitive_group=None, gaps=None, **kwargs):
        super().__init__( *args, class_dict, **kwargs)
        self.sensitive_group = sensitive_group
        self.class_dict = class_dict
        self.tpr_group_dict = {}
        self.classes_sorted = np.array([])
        self.gaps = gaps

    def _init_plots(self):
        '''
        Initialises matplotlib artists and plots.
        '''
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 16))
        self.ax.set_title(self.title)
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
                                       interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        # set the mouse-event listeners
        self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.set_axis_off()


    def compute_grid(self):
        '''
        Computes the visualisation of the decision boundaries.
        '''
        if self.verbose:
            print('Computing decision regions ...')
        # get extent of embedding
        x_min, y_min, x_max, y_max = self._get_plot_measures()
        # create grid
        xs = np.linspace(x_min, x_max, self.resolution)
        ys = np.linspace(y_min, y_max, self.resolution)
        self.grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(self.grid.reshape(self.grid.shape[0], -1), 0, 1)

        # map gridmpoint to images
        grid_samples = self.inverse(grid)

        mesh_preds = self._predict_batches(grid_samples)
        mesh_preds = mesh_preds + 1e-8

        self.mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(self.mesh_classes)

        self.classes_sorted = self.sort_classes()

        # Create a mapping from value to index in classes_sorted
        value_to_index = np.zeros(np.max(self.classes_sorted) + 1, dtype=int)
        value_to_index[self.classes_sorted] = np.arange(len(self.classes_sorted))

        # Map mesh values to their corresponding indices in classes_sorted
        mesh_adapt = value_to_index[self.mesh_classes]

        # get color of gridpoints
        color = self.cmap(mesh_adapt / mesh_max_class)
        # scale colors by certainty
        h = -(mesh_preds * np.log(mesh_preds)).sum(axis=1) / np.log(self.n_classes)
        h = (h / h.max()).reshape(-1, 1)
        # adjust brightness
        h = np.clip(h * 1.2, 0, 1)
        color = color[:, 0:3]
        color = (1 - h) * (0.5 * color) + h * np.ones(color.shape, dtype=np.uint8)
        decision_view = color.reshape(self.resolution, self.resolution, 3)
        return decision_view

    def sort_classes(self):
        # gap_list = np.array(self.calculate_gaps(1,0))
        sorted_indices = np.argsort(self.gaps)

        # Sort gaps using the sorted indices
        # gap_list_sorted = gap_list[sorted_indices]

        # Permute classes according to the sorted indices of b
        classes_sorted = self.classes[sorted_indices]
        return classes_sorted

    def show_sample(self, event):
        '''
        Invoked when the user clicks on the plot. Determines the
        embedded or synthesised sample at the click location and
        passes it to the data_viz method, together with the prediction,
        if present a groun truth label and the 2D click location.
        '''

        # when there is an artist attribute, a
        # concrete sample was clicked, otherwise
        # show the according synthesised image

        if event.key == "enter":
            indices = self.selector.ind
            sample, p, t = self.get_artist_sample(indices)
            # title = '%s <-> %s' if p != t else '%s --- %s'
            # title = title % (self.classes[p], self.classes[t])
        # elif not self.disable_synth:
        #     # workaraound: inverse embedding needs more points
        #     # otherwise it doens't work --> [point]*5
        #     point = np.array([[event.xdata, event.ydata]] * 5)
        #
        #     # if the outside of the plot was clicked, points are None
        #     if None in point[0]:
        #         return
        #
        #     sample = self.inverse(point)[0]
        #     sample += abs(sample.min())
        #     sample /= sample.max()
        #     # title = 'Synthesised at [%.1f, %.1f]' % tuple(point[0])
        #     p, t = self.get_mesh_prediction_at(*point[0]), None
        # else:
        #     self.disable_synth = False
        #     return

            if self.data_viz is not None:
                self.data_viz(sample, p, t, self.cmap)
            return
        else:
            warnings.warn("Data visualization not possible, as the partnet_grasp points have"
                          "no image shape. Pass a function in the data_viz argument,"
                          "to enable custom partnet_grasp visualization.")
            return
    def show(self):
        '''
        Shows the current plot.
        '''
        if not hasattr(self, 'fig'):
            self._init_plots()

        x_min, y_min, x_max, y_max = self._get_plot_measures()

        self.cls_plot.set_data(self.classifier_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
        desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
        self.desc.set_text(desc)

        scatter_plots = []
        labels = []

        for g in np.unique(self.sensitive_group):
            sensitive_g_filter = self.sensitive_group == g
            for index, cl in enumerate(self.classes_sorted):
                data = self.embedded[sensitive_g_filter][self.y_true[sensitive_g_filter] == cl]
                color = self.cmap(index / (self.n_classes - 1))
                if g == 0:
                    scatter = self.ax.scatter(data[:, 0], data[:, 1], color=color)

                    # Only add to legend for the first group to avoid duplicates
                    scatter_plots.append(scatter)
                    labels.append(self.class_dict[cl])
                else:
                    scatter = self.ax.scatter(data[:, 0], data[:, 1], marker=(5, 2), color=color)

        # Create legend for classes
        legend1 = self.ax.legend(scatter_plots, labels, loc="upper left", bbox_to_anchor=(1.04, 1), title="Classes")
        self.ax.add_artist(legend1)

        # Create dummy scatter plots for the gender legend
        male_scatter = self.ax.scatter([], [], marker='o', color='black', label='Male')
        female_scatter = self.ax.scatter([], [], marker=(5, 2), color='black', label='Female')

        # Create legend for gender
        legend2 = self.ax.legend(handles=[male_scatter, female_scatter], loc="upper right", title="Gender")
        self.ax.add_artist(legend2)

        # Adjust subplot parameters to make room for the legends
        plt.subplots_adjust(right=0.45)

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.selector = SelectFromCollection(self.ax, self.embedded)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()
