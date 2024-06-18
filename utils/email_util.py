# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2023-10-11 10:58
@Author  : lijing
@File    : email_util.py
@Description: 邮件工具类
---------------------------------------
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailServer:
    def __init__(
        self, smtp_server, smtp_port, sender_email, sender_password, receiver_email
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email

    def send_email(self, subject, message):
        """
        发送邮件
        :param sender_email: 发送者邮箱
        :param sender_password: 发送者密码
        :param receiver_email: 接收者邮箱
        :param subject: 主题
        :param message: 邮件消息
        :return: 返回值
        """

        # 创建邮件内容和头部
        email = MIMEMultipart()
        email["From"] = self.sender_email
        email["To"] = self.receiver_email
        email["Subject"] = subject

        # 添加邮件正文
        email.attach(MIMEText(message, "html"))

        # 建立SMTP连接并发送邮件
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(email)

    def send_email_2_admin(self, subject, message):
        """
        发送邮箱给管理员
        :param subject: 邮件主题
        :param message: 邮件信息
        :return:
        """
        # 设置邮件主题和内容，发送邮件
        # 发送邮件
        email_template = """
            <div class="email-content" style="width:90%;max-width:660px;margin:20px auto 30px;line-height:1.5;text-align:left;overflow:hidden;border-radius:8px;box-shadow:0 2px 12px 0 rgba(0,0,0,0.1);">
            <div style="overflow:hidden;">
              <h1 class="email-title" style="position:relative;margin:0;-webkit-box-sizing:border-box;box-sizing:border-box;padding:14px 52px 14px 20px;line-height:1.6;font-size:16px;font-weight:normal;color:#fff;background:-webkit-linear-gradient(-45deg,rgba(9,69,138,0.2),rgba(68,155,255,0.7),rgba(117,113,251,0.7),rgba(68,155,255,0.7),rgba(9,69,138,0.2));background:linear-gradient(-45deg,rgba(9,69,138,0.2),rgba(68,155,255,0.7),rgba(117,113,251,0.7),rgba(68,155,255,0.7),rgba(9,69,138,0.2));background-size:400% 400%;background-position:50% 100%;">
                Dear, 您有新的邮件消息！
              </h1>
              <div class="email-text" style="padding:20px 28px 10px;background:#fff;">
                <p style="margin:5px 0 5px;padding:0;line-height:24px;font-size:13px;color:#6e6e6e;"><span style="font-weight:bold;color:#9f98ff">筱晶哥哥，</span> 您好!</p>
                <div style="margin:12px 0;padding:18px 20px;white-space:pre-line;word-break:break-all;color:#6e6e6e;font-size:13px;background:#f8f8f8;background:repeating-linear-gradient(145deg, #f2f6fc, #f2f6fc 15px, #fff 0, #fff 25px);">{}</div>
               </div>
              <div class="email-footer" style="padding:10px 20px;border-top:1px solid #eee;">
                <p style="margin:0;padding:0;line-height:24px;font-size:13px;color:#999;">* 注意：此邮件由 <a href="javascript:;" style="color:#9f98ff;text-decoration:none;" rel="noopener">筱晶哥哥配置的邮件程序</a> 自动发出，请勿回复，如有打扰，请见谅。</p>
              </div>
            </div>
          </div>
            """
        self.send_email(subject, email_template.format(message))
