import numpy as np
import collections as cl

class Neuron(object):
    """
    The superclass of all WiSARD neurons. Some of the
    methods are abstract and have to be overridden.
    """

    def __init__(self, address_length=None):
        """
        Initializes the neuron.
        """
        self.locations = []
        self.address_length = address_length
        if address_length is None:
            self.max_entries = None
        else:
            self.max_entries = 2**address_length

    def __len__(self):
        """
        Returns how many RAM locations are written.
        """
        return len(self.locations)

    def record(self, address):
        """
        Records the given address.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def is_set(self, address):
        """
        Returns true iff the location being addressed is written.
        """
        return address in self.locations

    def bleach(self, threshold):
        """
        Bleaches RAM entries based on a defined model.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def remove(self, address):
        """
        Deletes the given address out of the RAM.
        """
        try:
            del self.locations[address]
        except BaseException:
            pass

    def clear(self):
        """
        Clears all RAM locations.
        """
        self.locations.clear()

    def intersection_level(self, neuron):
        """
        Returns the intersection level between two neurons.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def merge(self, neuron):
        """
        Merges the given neuron's knowledge into this one.
        """
        raise NotImplementedError("This method is abstract. Override it.")

# class DictNeuron(Neuron):
#     '''
#     Basic neuron based on dict(). Default neuron
#     counts how often an address was written?
#     '''
#     def __init__(self, type_=int):
#         # Address length? What is it used for?
#         self.locations = cl.defaultdict(type_)

#     def __len__(self):
#         return len(self.locations)
    
#     def record(self, address, intensity=1):
#         # self.locations[address] += intensity

#         # What's the point? Figure this section out
#         if address in self.locations:
#             self.locations[address] += intensity
#         else:
#             self.locations[address] = intensity

#     def is_set(self, address):
#         return address in self.locations

#     def count(self, address):
#         # returns the count else returns 0
#         return self.locations.get(address, 0)

#     def intersection_level(self, neuron):
#         '''
#         if a & b are intersection of locations written in both neurons and a | b their union,
#         this method returns (a & b)/(a | b).
#         '''
#         len_intersect = len(self.locations.keys() & neuron.locations.keys())
#         len_union = len(self.locations.keys() | neuron.locations.keys())
#         return len_intersect / len_union
    
#     def bleach(self, threshold):
#         '''
#         if location written more then threshold times, reduce by threshold
#         Otherwise, delete location
#         return number of deleted addresses
#         '''
#         count = 0
#         for address in list(self.locations.keys()):
#             if self.locations[address] > threshold:
#                 self.locations[address] -= threshold
#             else:
#                 del self.locations[address]
#                 count += 1
#         return count

#     def merge(self, neuron):
#         '''
#         The other guy implemented a 'merge' function. Should I?
#         '''
#         for address in neuron.keys():
#             self.locations[address] += neuron.locations[address]

class SWNeuron(Neuron):
    """
    This neuron uses a sliding window model and
    saves the last time an address was written.
    """

    def __init__(self, address_length=None):
        super().__init__(address_length)
        self.locations = cl.OrderedDict()

    def record(self, address, time):
        """
        Records the given address.
        """
        try:
            del self.locations[address]
        except KeyError:
            pass
        self.locations[address] = time

    def bleach(self, threshold):
        """
        Clears the locations recorded before the time threshold.
        Returns the number of deleted addresses.
        """
        count = 0
        for address, time in zip(list(self.locations.keys()),
                                 list(self.locations.values())):
            if time < threshold:
                del self.locations[address]
                count += 1
            else:
                break  # end because Dict is orderd
        return count

    def intersection_level(self, neuron):
        """
        Returns the amount of locations written in both neurons.

        Considering a & b the intersection of the locations written in both
        neurons and a | b their union, this method returns (a & b)/(a | b).
        """
        len_intrsctn = float(
            len(self.locations.keys() & neuron.locations.keys()))
        len_union = float(len(self.locations.keys() | neuron.locations.keys()))
        return len_intrsctn / len_union

    def merge(self, neuron):
        """
        Merges the given neuron's knowledge into this one.
        """
        intersection = set(
            self.locations.keys()) & set(
            neuron.locations.keys())
        self.locations = {**self.locations, **neuron.locations}
        # Need to differentiate regarding most recent timestamp
        if len(intersection) != 0:
            for address in intersection:
                self.locations[address] = max(
                    self.locations[address], neuron.locations[address])