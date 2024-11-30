from typing import *
from ipywidgets import IntSlider, BoundedIntText, HBox
from fastplotlib import LinearSelector
import numpy as np


class ComponentStore:
    def __init__(self, time_store):
        """
        Initialize the ComponentStore.

        Parameters
        ----------
        time_store : TimeStore
            Instance of TimeStore to handle synchronization across components.
        """
        self.time_store = time_store
        self._component_index = 0
        self.subscribers = []

    @property
    def component_index(self) -> int:
        """Get the current component index."""
        return self._component_index

    @component_index.setter
    def component_index(self, index: int):
        """Set the component index and update all subscribers."""
        self._component_index = index
        self._update_subscribers()

    def add_subscriber(self, subscriber: IntSlider | BoundedIntText | LinearSelector):
        """
        Add a subscriber to be updated when the component index changes.

        Parameters
        ----------
        subscriber : IntSlider | BoundedIntText | LinearSelector
            A UI widget or selector to synchronize with the component index.
        """
        if subscriber not in self.subscribers:
            self.subscribers.append(subscriber)

        if isinstance(subscriber, (IntSlider, BoundedIntText)):
            subscriber.observe(self._sync_with_widget, "value")
        elif isinstance(subscriber, LinearSelector):
            subscriber.add_event_handler(self._sync_with_selector, "selection")

    def _sync_with_widget(self, change):
        """Sync the component index with IntSlider or BoundedIntText."""
        self.component_index = change["new"]

    def _sync_with_selector(self, event):
        """Sync the component index with a LinearSelector."""
        self.component_index = int(event.info["value"])

    def _update_subscribers(self):
        """Update all subscribers with the current component index."""
        for subscriber in self.subscribers:
            if isinstance(subscriber, (IntSlider, BoundedIntText)):
                subscriber.value = self.component_index
            elif isinstance(subscriber, LinearSelector):
                subscriber.selection = self.component_index

    def remove_subscriber(self, subscriber):
        """
        Remove a subscriber from the store.

        Parameters
        ----------
        subscriber : IntSlider | BoundedIntText | LinearSelector
            The subscriber to be removed.
        """
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)

