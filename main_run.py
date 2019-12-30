'''
This script is used to run the the pipeline of MemGuard. 
'''
import os 
import configparser

import datetime
import logging
import logging.config

logging.config.fileConfig('log.ini')
logger = logging.getLogger('api')

config = configparser.ConfigParser()
config.read('config.ini')
#result_folder="../result/location/code_publish/"

"""
#train user_classification_model attack_shadow_mode defense_model
#for dataset in ["location","CH-MNIST","MNIST"]:
for dataset in ["MNIST"]:
    
    result_folder="result/"+dataset+"/code_publish/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    config[dataset]["result_folder"]=result_folder
    with open("config.ini",'w') as configfile:
        config.write(configfile)
        configfile.close()

    cmd="python train_user_classification_model.py -dataset "+dataset
    os.system(cmd)
    
    cmd="python train_user_classification_model.py -dataset "+dataset+" -l2"
    os.system(cmd)

    cmd="python train_defense_model_defensemodel.py  -dataset "+dataset
    os.system(cmd)

    cmd="python train_attack_shadow_model.py  -dataset "+dataset+" -adv adv1"
    os.system(cmd)

"""

#"""    
#compare diffrent defense and attack
#for dataset in ["location","CH-MNIST","MNIST"]:
for dataset in ["location"]:
    for attack in ["our","nn","rf"]:
    #for attack in ["nn"]:
        #for defense in ["MemGuard","L2"]:
        #for defense in ["MemGuard"]:
        for defense in ["L2"]:
            
            """
            if defense == "MemGuard":
                cmd="python defense_framework.py -dataset "+dataset+" -qt evaluation " 
                os.system(cmd)
            if defense == "None":
                cmd="python defense_None_framework.py -dataset "+dataset+" -qt evaluation " 
                os.system(cmd)
            if defense == "L2":
                cmd="python defense_None_framework.py -dataset "+dataset+" -qt evaluation -l2" 
                os.system(cmd)
            """
            
            print("*"*110)
            print("[dataset={}] evaluate attack={} defense={}".format(dataset,attack,defense))
            logger.debug("[dataset={}] evaluate attack={} defense={}".format(dataset,attack,defense))
            cmd="python evaluate_"+attack+"_attack.py -dataset "+dataset+" -scenario full -version v0 -defense "+defense
            #os.system(cmd)
            
#"""

"""           
#compare diffrent l2 and attack
for dataset in ["location"]:
#for dataset in ["CH-MNIST"]:    
    for attack in ["our","nn","rf"]:
    #for attack in ["our"]:
        for defense in ["L2"]:
            #for e in [0.0,0.005,0.010,0.015,0.020,0.025,0.030]:
            for e in [0.0,0.010,0.020,0.030,0.040,0.050]:
                print("*"*110)
                print("evaluate attack={} defense={} e={}".format(attack,defense,e))
                logger.debug("*"*110)
                logger.debug("[dataset={}]evaluate attack={} defense={} e={}".format(dataset,attack,defense,e)) 
                
                cmd="python train_user_classification_model.py -dataset {} -l2 -e {}".format(dataset,e)
                os.system(cmd)
                
                cmd="python defense_None_framework.py -dataset {} -qt evaluation -l2 -e {}".format(dataset,e) 
                os.system(cmd)
           
                
                cmd="python evaluate_our_attack.py -dataset {} -scenario full -version v0 -defense {} -e {} ".format(dataset,defense,e)
                os.system(cmd)                
""" 

#"""           
#compare diffrent dropout and attack
#for dataset in ["location"]:
for dataset in ["CH-MNIST"]:    
    for attack in ["our","nn","rf"]:
    #for attack in ["our"]:
        for defense in ["dropout"]:
            for e in [0.0,0.2,0.4,0.6,0.8,1.0]:
                print("*"*110)
                print("evaluate attack={} defense={} e={}".format(attack,defense,e))
                logger.debug("*"*110)
                logger.debug("[dataset={}]evaluate attack={} defense={} e={}".format(dataset,attack,defense,e)) 
                
                cmd="python train_user_classification_model.py -dataset {} -dropout -e {}".format(dataset,e)
                #os.system(cmd)
                
                cmd="python defense_None_framework.py -dataset {} -qt evaluation -dropout -e {}".format(dataset,e) 
                #os.system(cmd)
           
                cmd="python evaluate_our_attack.py -dataset {} -scenario full -version v0 -defense {} -e {} ".format(dataset,defense,e)
                #os.system(cmd)                
#""" 

"""           
#compare topN of our attack
#for dataset in ["location"]:
for dataset in ["CH-MNIST"]:
    for attack in ["our"]:
        for defense in ["MemGuard"]:
            for topn in [1,2,3,5,10]:
           
                print("*"*110)
                print("evaluate attack={} defense={} topn={}".format(attack,defense,topn))
                logger.debug("[dataset={}]evaluate attack={} defense={} topn={}".format(dataset,attack,defense,topn))
                #cmd="python evaluate_our_attack.py -dataset "+dataset+" -scenario full -version v0 -defense "+defense+" -topn "+topn
                cmd="python evaluate_our_attack.py -dataset {} -scenario full -version v0 -defense {} -topn {}".format(dataset,defense,topn)
                os.system(cmd)                
""" 


#"""           
#Impact of attack classifiers
for dataset in ["location","CH-MNIST"]:
#for dataset in ["CH-MNIST"]:
    for attack in ["our"]:
        for defense in ["MemGuard"]:
            for H in [2,3,4,5]:
           
                print("*"*110)
                print("evaluate attack={} defense={} H={}".format(attack,defense,H))
                logger.debug("[dataset={}]evaluate attack={} defense={} H={}".format(dataset,attack,defense,H))
                
                cmd="python evaluate_our_attack.py -dataset {} -scenario full -version v0 -defense {} -H {}".format(dataset,defense,H)
                os.system(cmd)                
#""" 