from typing import List
import numpy as np
from fastplotlib import ImageGraphic, LinearSelector, ScatterGraphic, ImageWidget, LineCollection
from ipywidgets import IntSlider, FloatSlider, BoundedIntText

from fastplotlib.graphics._features import FeatureEvent

MARGIN: float = 1


# TODO: need to make a method for automatic MARGIN setting based on the data


class StoreComponent:
    @property
    def subscriber(self) -> ImageGraphic | IntSlider | FloatSlider | LinearSelector | BoundedIntText:
        return self._subscriber

    @property
    def data(self) -> List[LineCollection]:
        return self._data

    def __init__(self, subscriber, data=None):
        """A component store for tracking and emmitting the currently-selected index."""
        self._subscriber = subscriber

        # must have data if ImageGraphic
        if isinstance(self.subscriber, (ImageGraphic, ScatterGraphic)):
            if not hasattr(data, 'shape'):
                raise ValueError("If passing in `ImageGraphic` must provide associated `ndarray` object to update " "data with.")
            self._data = data
        self._data_filter = data_filter


class ComponentStore:
    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, value: int | float):
        self._current_index = int(value)

    @property
    def store(self) -> List[StoreComponent]:
        return self._store

    def __init__(self):
        self._store = list()
        self._current_index = 0

    def subscribe(self,
                  subscriber: ImageWidget | ImageGraphic | LinearSelector | ScatterGraphic | IntSlider | FloatSlider,
                  data: np.ndarray = None,
                  data_filter: callable = None,
                  multiplier: int | float = None) -> None:
        """
        Method for adding a subscriber to the store to be synchronized.

        Parameters
        ----------
        subscriber: fastplotlib.ImageGraphic, fastplotlib.LinearSelector, ipywidgets.IntSlider, or ipywidgets.FloatSlider
            ipywidget or fastplotlib object to be synchronized
        data: np.ndarray, optional
            If subscriber is a fastplotlib.ImageGraphic, must have an associating numpy.ndarray to update data with.
        """
        # create a TimeStoreComponent
        component = ComponentStore(subscriber=subscriber)

        # add component to the store
        self._store.append(component)

        if isinstance(component.subscriber, ImageWidget):
            component.subscriber.add_event_handler(self._update_store, "current_index")
        if isinstance(component.subscriber, (IntSlider, FloatSlider)):
            component.subscriber.observe(self._update_store, "value")
        if isinstance(component.subscriber, LinearSelector):
            component.subscriber.add_event_handler(self._update_store, "selection")

    def unsubscribe(self, subscriber: ImageGraphic | LinearSelector | IntSlider | FloatSlider):
        for component in self.store:
            if component.subscriber == subscriber:
                #  remove the component from the store
                self.store.remove(component)
                # remove event handler
                if isinstance(component, (IntSlider, FloatSlider)):
                    component.unobserve(self._update_store)
                if isinstance(component, LinearSelector):
                    component.subscriber.remove_event_handler(self._update_store, "selection")

    def _update_store(self, ev):
        """Called when event occurs and store needs to be updated."""
        print("Updating store")
        # parse event to see if it originated from ipywidget or selector
        if isinstance(ev, FeatureEvent):
            # check for multiplier to adjust time
            for component in self.store:
                if isinstance(component.subscriber, LinearSelector):
                    if ev.graphic == component.subscriber:
                        self.current_index = ev.info["value"]
        elif isinstance(ev, dict):
            self.current_index = ev["index"]
        else:
            self.current_index = ev["new"]

        print('Iterating components')
        for component in self.store:
            print('Component 1')
            if isinstance(component.subscriber, ImageWidget):
                # user moved qslider, don't update imagewidget
                if isinstance(ev, dict) and 't' in ev:
                    pass
                else:
                    component.subscriber.current_index = {"t": self.time}
            elif isinstance(component.subscriber, ScatterGraphic):
                component.subscriber.data = component.data[self.time]
            # update ImageGraphic data no matter what
            elif isinstance(component.subscriber, ImageGraphic):
                if component.data_filter is None:
                    new_data = component.data[self.time]
                else:
                    new_data = component.data_filter(component.data[self.time])
                if new_data.shape != component.subscriber.data.value.shape:
                    raise ValueError(f"data filter function: {component.data_filter} must return data in the same shape" f"as the current data")
                component.subscriber.data = new_data
            elif isinstance(component.subscriber, LinearSelector):
                # only update if different
                if abs(component.subscriber.selection - (self.time * component.multiplier)) > MARGIN:
                    print('Is LinearSelector and abs(component.subscriber.selection - (self.time * '
                          'component.multiplier)) > MARGIN')
                    component.subscriber.selection = self.time * component.multiplier
            else:
                # only update if different
                if abs(component.subscriber.value - self.time) > MARGIN:
                    component.subscriber.value = self.time
