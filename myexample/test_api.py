import unittest
import warnings
import requests
import time, os
import json
import base64
import random
import string
import hashlib
import pickle
import sys
class PYreftTestCase(unittest.TestCase):
    """
    测试/opt/lang/retrivel/pyreft_api.py
    """
    host = '192.168.50.209'
    port = '7201'
    env_host = os.environ.get('host')
    if env_host:
        host = env_host
    env_port = os.environ.get('port')
    if env_port:
        port = env_port
    def test_ping(self):
        """
        测试Ping接口
        :return:
        """
        url = f"http://{self.host}:{self.port}/ping"
        #读取一条数据，GET模式，参数是一个字典
        start_time = time.time()
        # 提交form格式数据
        r = requests.get(url)
        assert r.json() == "Pong", f"接口是否未启动"
        print(f"花费时间: {time.time() - start_time}秒")
    def test_train_model_custom_data(self):
        """
        测试训练一个模型
        :return:
        {
            "code": 0,
            "data": [
                10,
                2.688475799560547,
                {
                    "epoch": 10.0,
                    "total_flos": 0.0,
                    "train_loss": 2.688475799560547,
                    "train_runtime": 2.3054,
                    "train_samples_per_second": 8.675,
                    "train_steps_per_second": 4.338
                }
            ],
            "msg": "success"
        }

        """
        url = f"http://{self.host}:{self.port}/api/train"
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        train_data = [
            ["Who am I?", "👤❓🔍🌟"],
            ["What's 2+2? And provide some details?", "🔢➕🔢➡️🍀"],
        ]
        data = {"train_data": train_data}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        assert r.status_code == 200, f"返回的status code不是200，请检查"
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        msg = res.get("msg")
        assert msg == "success", f"接口返回的msg不是成功，请检查"
        print(f"花费时间: {time.time() - start_time}秒")
    def test_train_model_all_data(self):
        """
        测试训练一个模型,使用所有训练数据
        :return:
        """
        url = f"http://{self.host}:{self.port}/api/train"
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        data = {"train_all": True}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        assert r.status_code == 200, f"返回的status code不是200，请检查"
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        msg = res.get("msg")
        assert msg == "success", f"接口返回的msg不是成功，请检查"
        print(f"花费时间: {time.time() - start_time}秒")
    def test_inference_model(self):
        """
        运行训练好的reft模型推理
        :return:
        """
        url = f"http://{self.host}:{self.port}/api/inference"
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        # question = "中国香氛市场过去一年消费者讨论香水的的变化趋势如何？"
        data = {"instruction": "who are you?", "specify_model_dir": "mind"}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        assert r.status_code == 200, f"返回的status code不是200，请检查"
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        msg = res.get("msg")
        assert msg == "success", f"接口返回的msg不是成功，请检查"
        print(f"花费时间: {time.time() - start_time}秒")
    def test_inference_model_orginal(self):
        """
        推理模型, 原始的模型推理
        :return:
        """
        url = f"http://{self.host}:{self.port}/api/inference"
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        # question = "中国香氛市场过去一年消费者讨论香水的的变化趋势如何？"
        data = {"instruction": "who are you?", "original_model": True}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        assert r.status_code == 200, f"返回的status code不是200，请检查"
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        msg = res.get("msg")
        assert msg == "success", f"接口返回的msg不是成功，请检查"
        print(f"花费时间: {time.time() - start_time}秒")
    def test_del_train_data(self):
        """
        删除suo yo
        :return:
        """
        url = f"http://{self.host}:{self.port}/api/train"
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        train_data = [
            ["Who am I?", "👤❓🔍🌟"],
            ["What's 2+2? And provide some details?", "🔢➕🔢➡️🍀"],
        ]
        data = {"train_data": train_data}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        assert r.status_code == 200, f"返回的status code不是200，请检查"
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        msg = res.get("msg")
        assert msg == "success", f"接口返回的msg不是成功，请检查"
        print(f"花费时间: {time.time() - start_time}秒")