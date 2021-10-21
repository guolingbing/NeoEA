import tensorflow as tf
import numpy as np
import pandas as pd



class NeuralAxiom(object):

    def __init__(self, args, kg, model):
        self.args = args
        self.kg = kg
        self.model = model
        self.ent_embeds = model.ent_embeds
        self.rel_embeds = model.rel_embeds
        
        self.ents = np.array(kg.entities_list)
        self.rels = np.array(kg.relations_list)
        
        self.ent_sample_size = self.sample_size = min(len(self.ents)//4, args.neo_sample_size)
        self.rel_sample_size = min(len(self.rels)//4, 4)
        
        self.init_graph()

    def sample(self, triples=None):
        pass

    def construct(self):
        pass
    
    def init_graph(self):
        pass

    
class BasicNeuralAxiom(NeuralAxiom):
    
    def sample(self, triples=None):
        entities = pd.concat([triples.h, triples.t], axis=0, ignore_index=True).values
        relations = triples.r.values
        
        sampled_ents = np.random.choice(entities, size=self.ent_sample_size, replace=False)
        sampled_rels = np.random.choice(relations, size=self.rel_sample_size, replace=False)
        
        return {self.sampled_ent_ph:sampled_ents, self.sampled_rel_ph:sampled_rels}
    

    
    def init_graph(self):
        with tf.name_scope('basic_axiom'):
            self.sampled_ent_ph = tf.placeholder(tf.int32, shape=[self.ent_sample_size,])
            self.sampled_rel_ph = tf.placeholder(tf.int32, shape=[self.rel_sample_size,])

            ent_embeds, rel_embeds = self.ent_embeds, self.rel_embeds
            
            
            feature_dists = []
            ent_feature_dist = tf.nn.embedding_lookup(ent_embeds, self.sampled_ent_ph)
            feature_dists.append(ent_feature_dist)
            
            if rel_embeds is not None:
                rel_feature_dist = tf.nn.embedding_lookup(rel_embeds, self.sampled_rel_ph)
                feature_dists.append(rel_feature_dist)

            self.feature_dists = feature_dists
        
        return self.feature_dists
    
class ECRNeuralAxiom(NeuralAxiom):
    '''
    Entity Conditioned on Relation Neural Axiom
    '''
    def sample(self, triples=None):
        # first sample relation candidates
        sampled_indices = np.random.choice(len(triples), size=self.ent_sample_size, replace=False)
        
        sampled_triples = triples.iloc[sampled_indices].values
        
        return {self.sampled_ecr_tripels:sampled_triples}
    
    def con_ops(self, ent, rel):
        if self.args.neo_condition_method == 'multiply':
            return ent * rel
        elif self.args.neo_condition_method == 'concat':
            return tf.concat([ent,rel], axis=-1)
        elif self.args.neo_condition_method == 'add':
            return tf.nn.relu(tf.layer.dense(ent, ent.shape[-1], use_bias=False) + tf.layer.dense(rel, ent.shape[-1], use_bias=False))

    def init_graph(self):
        with tf.name_scope('ECR_axiom'):
            self.sampled_ecr_tripels = tf.placeholder(tf.int32, shape=[self.ent_sample_size,3])

            ent_embeds, rel_embeds = self.ent_embeds, self.rel_embeds
            
#             kernel = tf.get_variable
            
            
            feature_dists = []
            
            head, rel, tail = tf.nn.embedding_lookup(ent_embeds, self.sampled_ecr_tripels[:, 0]),\
                tf.nn.embedding_lookup(rel_embeds, self.sampled_ecr_tripels[:, 1]),\
                tf.nn.embedding_lookup(ent_embeds, self.sampled_ecr_tripels[:, 2]),
            
            hcr = self.con_ops(head, rel)
            tcr = self.con_ops(tail, rel)
            
            feature_dists += [hcr, tcr]

            self.feature_dists = feature_dists
        
        return self.feature_dists

class TCRNeuralAxiom(NeuralAxiom):
    '''
    Triple Conditioned on Relation Neural Axiom
    '''
    def sample(self, triples=None):
        # first sample relation candidates
        sampled_indices = np.random.choice(len(triples), size=self.ent_sample_size, replace=False)
        
        sampled_triples = triples.iloc[sampled_indices].values
        
        return {self.sampled_ecr_tripels:sampled_triples}


    def con_ops(self, head, tail, rel):
        if self.args.neo_condition_method == 'multiply':
            return head * rel * tail
        elif self.args.neo_condition_method == 'concat':
            return tf.concat([head,rel,tail], axis=-1)
        elif self.args.neo_condition_method == 'add':
            return tf.nn.relu(tf.layer.dense(head, head.shape[-1], use_bias=False) +
             tf.layer.dense(rel, ent.shape[-1], use_bias=False) +
              tf.layer.dense(tail, ent.shape[-1], use_bias=False))
        
    def init_graph(self):
        with tf.name_scope('ECR_axiom'):
            self.sampled_ecr_tripels = tf.placeholder(tf.int32, shape=[self.ent_sample_size,3])

            ent_embeds, rel_embeds = self.ent_embeds, self.rel_embeds
            
            
            feature_dists = []
            
            head, rel, tail = tf.nn.embedding_lookup(ent_embeds, self.sampled_ecr_tripels[:, 0]),\
                tf.nn.embedding_lookup(rel_embeds, self.sampled_ecr_tripels[:, 1]),\
                tf.nn.embedding_lookup(ent_embeds, self.sampled_ecr_tripels[:, 2]),
            
            tcr = self.con_ops(head, rel, tail)
            
            feature_dists += [tcr,]

            self.feature_dists = feature_dists
        
        return self.feature_dists


class NeuralOntology(object):

    def __init__(self, args, kg, model):
        args.neo_sample_size = args.batch_size
        self.args = args
        self.kg = kg
        self.model = model
        
        self.axioms = [BasicNeuralAxiom(args, kg, model), ECRNeuralAxiom(args, kg, model), TCRNeuralAxiom(args, kg, model)]
        print('init neural axioms:')
        print (self.axioms)
        self.onto_dist = [axiom.feature_dists for axiom in self.axioms]
        
        self.triples =  pd.DataFrame(kg.relation_triples_list, columns=['h','r','t'])

    def collect(self):
        sampled_indices = np.random.choice(len(self.triples), size=self.args.neo_sample_size, replace=False)
        sampled_triples = self.triples.iloc[sampled_indices]
        
        ontology_sampled_input = {}
        for axiom in self.axioms:
            ontology_sampled_input.update(axiom.sample(sampled_triples))
        
        return ontology_sampled_input



class NeuralOntologyAlignment(object):
    
    def __init__(self, args, kgs, model):
        args.neoea_method = 'wd'
        
        self.args = args
        self.kgs = kgs
        self.model = model
        
        self.init_graph()
        
        
    def collect(self):
        neo_input1 = self.neo1.collect()
        neo_input2 = self.neo2.collect()
        
        return {**neo_input1, **neo_input2}
    
    
    
    def wasserstein_distance(self, dist1, dist2):
        batch_size= int(dist1.shape[0])
        input_dim = int(dist1.shape[-1])

        def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense'):
            with tf.name_scope(layer_name):
                weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')
                bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
                if input_type == 'sparse':
                    activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
                else:
                    activations = act(tf.matmul(input_tensor, weight) + bias)
                return activations
        
        def critic(dist):
            critic_h1 = fc_layer(dist, input_dim, 100, layer_name='critic_h1')
            out = fc_layer(critic_h1, 100, 1, layer_name='critic_h2', act=tf.identity)
            return out
        
        offset = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        interpolates = dist2 + (dist1-dist2)*offset

        concated = tf.concat([dist1, dist2, interpolates], axis=0)

        critic_out = critic(concated)
        critic_dist1, critic_dist2 = critic_out[:batch_size], critic_out[batch_size:batch_size*2]
        wd_loss = tf.reduce_mean(critic_dist1-critic_dist2)
        
        critic_grads = tf.gradients(critic_out, [concated,])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(critic_grads), axis=1))
        grad_penalty = tf.reduce_mean((slopes-1)**2)
        
        
        embed_loss = wd_loss
        
        metric_loss = -wd_loss+10*grad_penalty
        
        return embed_loss, metric_loss
        
    def init_graph(self):
        args, kgs, model = self.args, self.kgs, self.model

        with tf.name_scope('neo_ontology'):
            self.neo_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.neo1 = NeuralOntology(args, kgs.kg1, model)
            self.neo2 = NeuralOntology(args, kgs.kg2, model)
            
            neo_dist1 = self.neo1.onto_dist
            neo_dist2 = self.neo2.onto_dist
            
            metrics = {'wd':self.wasserstein_distance,}
            self.metric = metrics[self.args.neoea_method]
            
            embed_losses, metric_losses = [], []
            for axiom1, axiom2 in zip(neo_dist1, neo_dist2):
                for dist1, dist2 in zip(axiom1, axiom2):
                    embed_loss, metric_loss = self.metric(dist1, dist2)
                    
                    embed_losses.append(embed_loss)
                    metric_losses.append(metric_loss)
            
            
        # final ops
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.metric_vars = [var for var in variables if 'neo' in var.name]
        self.kge_vars = [var for var in variables if var not in self.metric_vars]
        
        # for KGE model tries to minimize this loss to align the neural ontologies
        self.embed_loss = tf.reduce_sum(tf.reduce_mean(embed_losses, axis=0)) * self.args.neo_param

        
        
        # while the metrics tries to discriminate the two ontologies,
        # by maxmizing the loss
        self.metric_loss = tf.reduce_sum(tf.reduce_mean(metric_losses, axis=0))
        self.metric_op = self.neo_optimizer.minimize(self.metric_loss, var_list=self.metric_vars)
        
            
    def run_one_epoch(self, one_batch_input={}):
        
        sess = self.model.session
        epoch_loss = []
        for step in range(self.args.neo_steps):
            one_batch_input.update(self.collect())
            step_loss, _ = sess.run([self.metric_loss, self.metric_op], feed_dict=one_batch_input)
            epoch_loss.append(step_loss)
        
        return -np.mean(epoch_loss)
        
    
        

    