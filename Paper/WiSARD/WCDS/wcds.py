import collections as cl
# import utilities as util
# import encoding as encoding
from WiSARD.WCDS import discriminators as dscrm
import numpy as np
import sys
import random
import logging
import time
import operator
from collections import OrderedDict

# class WiSARDLikeClassifier(object):
#     '''
#     Superclass of WiSARD classifiers

#     This is used as a template to all classifier implementations as an abstract class
#     '''

#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError('Abstract class. Derive')

#     def record(self, observation, class_):
#         '''
#         records the provided observation and relate it to the given class
#         '''
#         raise NotImplementedError('Abstract method. Override')

#     def answer(self, observation, class_=None):
#         '''
#         Returns how similar observations is to each known class

#         A dictionary with class labels as keys and the similarities between
#         obersvations and classes as values is returned

#         Parameters
#             oberservation: observation in which answers are based
#             class_: if given, only similarities with refrence to this class returned
#             '''
#         raise NotImplementedError('Abstract method. Override')

#     def counts(self, observation, class_=None):
#         '''
#         Returns description of obervations similarities to known classes

#         Takes into account the number of times observation addresses were 
#         previously recorded

#         A dictionary with class labels as keys and their respective similarity
#         descriptions as values returned

#         Parameters
#             observations: observation which answers are based
#             class_: if given, only answer with reference to this class returned
#         '''
#         raise NotImplementedError('Abstract method. Override')

#     def remove_class(self, class_):
#         raise NotImplementedError('Abstract method. Override')


# class WiSARD(WiSARDLikeClassifier):
#     def __init__(self, addresser=None, discriminator_factory=dscrm.HitDiscriminator):
#         ''' 
#         Inits WiSARD classifier
#         Parameters:
#             discriminator: the type of discriminator used to learn about each
#             class present. Which one should I choose?
#             change discriminators.temp into something - 
#                 discriminators.FrequencyDiscriminators for example
#         '''
#         if addresser is None:
#             self.addresser = encoding.BitStringEncoder()
#         else:
#             self.addreser = addresser
#         # Creates a dictionary of "dicriminators"
#         self.discriminators = cl.defaultdict(discriminator_factory)
    
#     def clear(self):
#         self.discriminators.clear()
    
#     def fit(self, observations, classes, clear=True):
#         if clear:
#             self.clear()
        
#         # data = it.zip(it.imap(self.addresser, observations), classes)
#         # Replace above line with python 3 code
#         # Maps the binary encoder to each observation and zips it with the corresponding class
#         temp = list(zip(list(map(self.addresser, observations)), classes))
#         temp2 = temp[:]
#         data = list(temp2)

#         for observation, class_ in data:
#             self.discriminators[class_].record(observation)

#         return self
    
#     def answer(self, observation):
#         return {class_: discriminator.answer(self.addresser(observation))
#             for class_, discriminator in self.discriminators.items()}
    
#     def predict(self, observations):
#         # What is this actually returning?
#         return [util.ranked(self.answer(observation))[0][0]
#             for observation in observations]

#     def bleach(self, threshold):
#         for d in self.discriminators:
#             self.discriminators[d].bleach(threshold)

#     def remove_class(self, class_):
#         del self.discriminators[class_]

# # TODO: Find a discriminator to implement. But which one?
# # I think the default one for the bachelor implementation is the SWDiscrim

class WiSARD(object):
    """
    The superclass of all WiSARD-like classifiers.

    This should be used as a template to any implementation, as an
    abstract class, but no method is indeed required to be overridden.
    """

    def __init__(self):
        """
        Initializes the classifier.
        """
        raise NotImplementedError("This class is abstract. Derive it.")

    def __len__(self):
        """
        Returns the number of discriminators in the classifier.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def clear(self):
        """
        Clears the discriminators.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def record(self, observation):
        """
        Records the provided observation.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def save(self, path):
        """
        Saves the classifier.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def load(self, path):
        """
        Loads the classifier.
        """
        raise NotImplementedError("This method is abstract. Override it.")


class WCDS(WiSARD):
    '''
    Implements WCDS
    '''
    def __init__(self, omega, delta, gamma, epsilon, dimension, beta=None, mu=0,
                    discriminator_factory=dscrm.SWDiscriminator, mapping="random", seed=random.randint(0, sys.maxsize)):
        
        '''
        Constructor for WCDS

        Parameters
        ----------
            omega
            delta
            gamme
            epsilon
            dimension
            beta
            mu
            discriminator_factory
            mapping
            seed
        '''

        if beta is None:
            self._adjust_parameters(gamma, dimension, delta)
        else:
            self.beta = beta
            self.delta = delta
            self.gamma = gamma
            if ((beta * delta) / (gamma * dimension)) < 1:
                self._adjust_parameters(gamma, dimension, delta)
            elif not ((beta * delta) / (gamma * dimension)).is_integer():
                self._adjust_parameters(gamma, dimension, delta)
        self.omega = omega
        self.epsilon = epsilon
        self.mu = mu
        self.dimension = dimension
        self.mapping = mapping 
        self.seed = seed 
        self.discriminator_factory = discriminator_factory
        self.discriminators = OrderedDict()
        self.discriminator_id = 0 # ?
        self.LRU = OrderedDict()
        logging.info(
                "Initialized WCDS with:\n Omega {}\n Delta {}\n Gamma {}\n Beta {}\n Epsilon {}\n Mu {}\n Dimension {}\n Seed {}\n {} mapping".format(
                    self.omega,
                    self.delta,
                    self.gamma,
                    self.beta,
                    self.epsilon,
                    self.mu,
                    self.dimension,
                    self.seed,
                    self.mapping))

    def __len__(self):
        return len(self.discriminators)

    def _adjust_parameters(self, gamma, dimension, delta):
        '''
        Have to fulfil property gamma * dimension = delta * beta
        '''

        if delta == gamma:
            self.gamma = gamma
            self.delta = delta
            self.beta = dimension
            return
        if dimension * gamma == delta:
            self.gamma = gamma 
            self.delta = delta
            self.beta = 1
            return
        if dimension * gamma < delta:
            self.gamma = int(round((1. * delta) / dimension))
            self.delta = dimension * self.gamma
            self.beta = 1
            return
        if (dimension * 1. * gamma / delta).is_integer():
            self.beta = int(dimension * gamma / delta)
            self.gamma = gamma
            self.delta = delta
        else:
            self.beta = dimension * gamma // delta
            self.delta = delta
            while not (delta * self.beta / dimension).is_integer():
                self.beta += 1
            self.gamma = int(delta * self.beta / dimension)
    
    def record(self, observation, time):
        """
        Absorbs observation and timestamp.
        Returns the id of the discriminator
        that absorbed the observation and a
        id list of deleted discriminators.
        """
        for i in observation:
            if round(i, 5) > 1 or i < 0:
                raise ValueError(
                    "Feature of given instance {} is not in [0:1]!".format(observation))
        logging.info(
            "Received: Observation: {} Time: {}".format(
                observation, time))

        # Delete outdated information
        lru_iter = iter(list(self.LRU.keys()))
        deleted_addr = 0
        try:
            current = next(lru_iter)
        except StopIteration:
            current = None
        if current is not None:
            while self.LRU[current] < time - self.omega:
                k, j, a = current
                del self.discriminators[k].neurons[j].locations[a]
                del self.LRU[current]
                deleted_addr += 1
                try:
                    current = next(lru_iter)
                except StopIteration:
                    break
        logging.info("Deleted {} outdated addresses.".format(deleted_addr))

        # Delete useless discriminators
        deleted_discr = self.clean_discriminators()
        logging.info(
            "Deleted {} empty discriminators.".format(
                len(deleted_discr)))

        # Calculate addressing of the observation
        addressing = self.addressing(observation)
        logging.info("Calculated addressing: {}".format(addressing))

        # Check if there is at least one discriminator
        if len(self.discriminators) > 0:
            # Calculate id and matching of best fitting discriminator
            k, best_matching = self.predict(addressing)
            logging.info(
                "Best discriminator {} matches {}%".format(
                    k, best_matching * 100))

            # If matching is too small create new discriminator
            if best_matching < self.epsilon:
                logging.info(
                    "Matching is too small. Creating new discriminator with id {}".format(
                        self.discriminator_id))
                d = self.discriminator_factory(
                    self.delta, self.discriminator_id)
                self.discriminators[self.discriminator_id] = d
                k = self.discriminator_id
                self.discriminator_id += 1
        else:
            # No discriminator yet
            logging.info(
                "No discriminators - creating a new one with id {}".format(self.discriminator_id))
            d = self.discriminator_factory(self.delta, self.discriminator_id)
            self.discriminators[self.discriminator_id] = d
            k = self.discriminator_id
            self.discriminator_id += 1

        # Absorb the current observation
        self.discriminators[k].record(addressing, time)
        for i, address in enumerate(addressing):
            # Delete to keep dict ordered by time
            try:
                del self.LRU[(k, i, address)]
            except KeyError:
                pass
            self.LRU[(k, i, address)] = time
        logging.info(
            "Absorbed observation. Current number of discriminators: {}".format(len(self)))

        return k, deleted_discr
    
    def predict(self, addressing):
        '''
        matches the best discriminator
        if two returns the same matching, the first one is taken
        '''

        predictions = [(d, self.discriminators[d].matching(addressing, self.mu)) for d in self.discriminators]
        if len(predictions) == 1:
            return predictions[0]
        predictions.sort(key=lambda x: x[1])
        k, best_matching = predictions[-1]
        confidence = 0
        if predictions[-1][1]:
            confidence = (predictions[-1][1] - predictions[-2][1]) / predictions[-1][1]
        logging.info(
            "Prediction has a confidence of {}%".format(confidence * 100)
        )
        return k, best_matching
    
    def addressing(self, observation):
        '''
        calculate and return the addressing for a given observation
        '''
        binarization = np.array([self._binarize(x_i, self.gamma) for x_i in observation])

        if self.mapping == "linear":
            binarization = binarization.flatten()
            addressing = []
            for i in range(self.beta * self.delta):
                addressing.append(binarization[i % len(binarization)])
            addressing = np.reshape(addressing, (self.delta, self.beta))
            addressing = [tuple(b) for b in addressing]
            return addressing
        elif self.mapping == "random":
            binarization = binarization.flatten()
            mapping = list(range(self.beta * self.delta))
            random.seed(self.seed)
            random.shuffle(mapping)
            addressing = np.empty(len(mapping))
            for i, m in enumerate(mapping):
                addressing[m] = binarization[i % len(binarization)]
            addressing = np.reshape(addressing, (self.delta, self.beta))
            addressing = [tuple(b) for b in addressing]
            return addressing
        else:
            raise ValueError("Mapping has an invalid value!")
    
    def _binarize(self, x, resolution):
        b = list()
        b.extend([1 for _ in range(int(round(x * resolution)))])
        b.extend([0 for _ in range(resolution - len(b))])
        return b

    def clean_discriminators(self):
        deleted = []
        for k in list(self.discriminators.keys()):
            if not self.discriminators[k].is_useful():
                del self.discriminators[k]
                deleted.append(k)
        return deleted
    
    def bleach(self, threshold):
        count = 0
        for d in self.discriminators.values():
            count += d.bleach(threshold)
        return count

    def reverse_addressing(self, addressing):
        """
        Reverse the applied addressing
        and return an observation.
        """
        if self.mapping == "linear":
            # Undo mapping
            addressing = addressing.flatten()
            unmapped_matrix = []
            for i in range(self.dimension * self.gamma):
                unmapped_matrix.append(addressing[i])
            unmapped_matrix = np.array(unmapped_matrix).reshape(
                (self.dimension, self.gamma))

            # Calculate observation
            observation = []
            for bin_ in unmapped_matrix:
                observation.append(float(sum(bin_)) / self.gamma)

            return observation
        elif self.mapping == "random":
            # Undo random mapping
            addressing = addressing.flatten()
            mapping = list(range(len(addressing)))
            random.seed(self.seed)
            random.shuffle(mapping)
            unmapped_matrix = np.empty(self.dimension * self.gamma)
            for i, m in enumerate(mapping):
                unmapped_matrix[i %
                                (self.dimension * self.gamma)] = addressing[m]
            unmapped_matrix = unmapped_matrix.reshape(
                (self.dimension, self.gamma))

            # Calculate observation
            observation = []
            for bin_ in unmapped_matrix:
                observation.append(float(sum(bin_)) / self.gamma)

            return observation
        else:
            raise ValueError("Mapping has an invalid value!")
    
    def _bit_to_int(self, bits):
        """
        Returns integer for list of bits.
        WARNING: Does not handle bit strings longer than 32 bits.
        """
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        return out
    
    def centroid(self, discr):
        """
        Approximates the centroid of a given discriminator.
        To properly work, the discriminator should not have
        been affected by a split beforehand.
        """
        # TODO: Make this a discriminator function
        # Calculate the mapped matrix
        mapped_matrix = []
        for neuron in discr.neurons:
            mean_address = tuple([0 for _ in range(self.beta)])
            for loc in neuron.locations:
                mean_address = tuple(map(operator.add, mean_address, loc))
            if len(neuron.locations) > 0:
                mean_address = tuple([x / len(neuron.locations)
                                      for x in mean_address])
            mapped_matrix.append(mean_address)
        mapped_matrix = np.array(mapped_matrix)

        # Calculate the coordinates
        coordinates = self.reverse_addressing(mapped_matrix)

        return coordinates
    
    def drasiw(self, discr, sampling=None):
        """
        Returns a list of points representing the given discriminator.
        """
        # TODO: Make this a discriminator function
        points = set()
        if not sampling:
            sampling = max([len(neuron) for neuron in discr.neurons])

        # Sampling points as often as maximum neuron length or sampling
        for _ in range(sampling):
            # Retreive random adresses
            mapped_matrix = []
            for i in range(len(discr.neurons)):
                sample = list(
                    random.choice(
                        list(
                            discr.neurons[i].locations.keys())))
                mapped_matrix.extend(sample)
            mapped_matrix = np.reshape(mapped_matrix, (self.delta, self.beta))
            # Reverse addressing
            points.add(tuple(self.reverse_addressing(mapped_matrix)))
        return list(points)