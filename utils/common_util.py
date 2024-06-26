# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-06-20 10:19
@Author  : lijing
@File    : common_util.py
@Description: 公共工具类
---------------------------------------
"""

import zipfile
import sys
from config.logger import Logger

logger = Logger().get_logger


def count_lines(filename):
    """
    获取文件的总行数
    :param filename: 文件名称
    :return:
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = sum(1 for line in file)
    return lines


def print_banner():
    """
    打印banner
    :return:
    """
    logger.debug(
f"""
Python Version: {sys.version}
////////////////////////////////////////////////////////////////////
//                          _ooOoo_                               //
//                         o8888888o                              //
//                         88" . "88                              //
//                         (| ^_^ |)                              //
//                         O\  =  /O                              //
//                      ____/`---'\____                           //
//                    .'  \\\\|     |//  `.                         //
//                   /  \\\\|||  :  |||//  \                        //
//                  /  _||||| -:- |||||-  \                       //
//                  |   | \\\\\  -  /// |   |                       //
//                  | \_|  ''\---/''  |   |                       //
//                  \  .-\__  `-`  ___/-. /                       //
//                ___`. .'  /--.--\  `. . ___                     //
//              ."" '<  `.___\_<|>_/___.'  >'"".                  //
//            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
//            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
//      ========`-.____`-.___\_____/___.-`____.-'========         //
//                           `=---='                              //
//      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
//             佛祖保佑       永不宕机      永无BUG                  //
////////////////////////////////////////////////////////////////////""")


def extract_txt_from_zip(zip_file):
    """
    从压缩包中读取txt中的内容汇总到一个字符串列表中
    :param zip_file: zip文件数据
    :return: 字符串列表
    """
    file_contents = []
    file_zip = zipfile.ZipFile(zip_file, "r")
    for file_name in file_zip.namelist():
        if file_name.endswith(".txt"):
            txt_content = file_zip.read(file_name).decode("utf-8")
            file_contents.append(txt_content)
    return file_contents


def readBanner(banner_path):
    """
    读取banner信息
    :param banner_path:
    :return:
    """
    with open(banner_path, "r") as file:
        banner = file.read()
    return banner
