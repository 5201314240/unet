import streamlit as st
import cv2
import numpy as np
import os
from openai import OpenAI
import matplotlib.pyplot as plt
from keras.models import load_model
import re
from sklearn.preprocessing import MinMaxScaler
import time
scaler = MinMaxScaler()
from gpt4all import GPT4All


def upload_image():
    uploaded_file = st.file_uploader("请上传你的文件：", type=['jpg', 'png', 'tif', 'tiff'])

    keep_directory = './images/'
    os.makedirs(keep_directory, exist_ok=True)

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(keep_directory, file_name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"图片已成功保存到{file_path}")


def process_image():
    def file_selector(folder_path="./images"):
        filenames = os.listdir(folder_path)

        selected_file = st.selectbox("请选择你的文件", filenames, index=None)
        if selected_file is not None:
            return os.path.join(folder_path, selected_file)

    filepath = file_selector()

    if filepath is not None:

        if st.button("查看图片"):
            st.image(filepath)

        image = cv2.imread(filepath)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x = st.slider('Change Threshold value', min_value=50, max_value=255)
        ret, thresh1 = cv2.threshold(imgray, x, 255, cv2.THRESH_BINARY)
        thresh1 = thresh1.astype(np.float64)
        st.image(thresh1, use_column_width=True, clamp=True)

        st.text("Bar Chart of the image")
        histr = cv2.calcHist([imgray], [0], None, [256], [0, 256])
        st.bar_chart(histr)

        st.text("Press the button below to view Canny Edge Detection Technique")
        if st.button('Canny Edge Detection'):
            edges = cv2.Canny(imgray, 50, 300)
            # cv2.imwrite('edges.jpg',edges)
            st.image(edges, use_column_width=True, clamp=True)

        y = st.slider("Change Value to increase or decrease contours", min_value=50, max_value=255)
        if st.button('Contours'):
            image = cv2.imread(filepath)
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, y, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            st.image(thresh, use_column_width=True, clamp=True)
            st.image(img, use_column_width=True, clamp=True)


def model_predict(model):
    global pred

    def file_selector(folder_path="./images"):
        filenames = os.listdir(folder_path)

        selected_file = st.selectbox("请选择你的图像", filenames, index=None)
        if selected_file is not None:
            return os.path.join(folder_path, selected_file)

    filepath = file_selector()

    if filepath is not None:

        image = cv2.imread(filepath)
        images = unit_image_process(image)
        plt.imshow(image, cmap='viridis')
        plt.axis('off')  # 不显示坐标轴
        st.pyplot(plt)

        selected_model = st.selectbox("请选择你的模型", ['model1', 'model2'])

        if selected_model == 'model1':
            pred = model.predict(images)
            pred = np.argmax(pred, axis=3)
            pred = pred[0, :, :]
            pred = label_to_rgb(pred)
        elif selected_model == 'model2':
            pred = model.predict(images)
            pred = np.argmax(pred, axis=3)
            pred = pred[0, :, :]
            pred = label_to_rgb(pred)

        plt.imshow(pred, cmap='viridis')
        plt.axis('off')  # 不显示坐标轴
        st.pyplot(plt)

        st.markdown("#### 两种模型的损失函数对比")
        # 使用 Streamlit 创建列布局
        col1, col2 = st.columns(2)

        # 在第一列中显示第一个图像
        with col1:
            plt.imshow(cv2.imread('data/history1_loss.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

            # 在第二列中显示第二个图像
        with col2:
            plt.imshow(cv2.imread('data/history2_loss.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

        st.markdown("#### 两种模型的交并比对比")
        col1, col2 = st.columns(2)

        # 在第一列中显示第一个图像
        with col1:
            plt.imshow(cv2.imread('data/history1_IoU.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)

            # 在第二列中显示第二个图像
        with col2:
            plt.imshow(cv2.imread('data/history2_IoU.jpg'), cmap='viridis')
            plt.axis('off')  # 不显示坐标轴
            st.pyplot(plt)


def label_to_rgb(predicted_image):
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i + 2], 16) for i in (0, 2, 4)))  # 60, 16, 152

    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i + 2], 16) for i in (0, 2, 4)))  # 132, 41, 246

    Road = '#6EC1E4'.lstrip('#')
    Road = np.array(tuple(int(Road[i:i + 2], 16) for i in (0, 2, 4)))  # 110, 193, 228

    Vegetation = 'FEDD3A'.lstrip('#')
    Vegetation = np.array(tuple(int(Vegetation[i:i + 2], 16) for i in (0, 2, 4)))  # 254, 221, 58

    Water = 'E2A929'.lstrip('#')
    Water = np.array(tuple(int(Water[i:i + 2], 16) for i in (0, 2, 4)))  # 226, 169, 41

    Unlabeled = '#9B9B9B'.lstrip('#')
    Unlabeled = np.array(tuple(int(Unlabeled[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))

    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled

    segmented_img = segmented_img.astype(np.uint8)
    return (segmented_img)


def unit_image_process(images):
    patch_size = 256
    image = cv2.resize(images, (patch_size, patch_size))
    single_patch_img = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    single_patch_img = np.expand_dims(single_patch_img, axis=0)
    return single_patch_img

def chatbot():
    # model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # downloads / loads a 4.66GB LLM

    def word_split(sentence):
        words = re.split(r"\W+", sentence)
        filtered_words = [word for word in words]
        return filtered_words

    if "messages" not in st.session_state:
        st.session_state.messages = [{'role': 'assistant', 'content': '你好，你有什么想问的嘛？'}]

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html=True)

    if prompt := st.chat_input("请输入您的信息"):
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    if prompt:
        def generate_stream(data):
            for char in data:
                yield char
                time.sleep(0.1)

        my_dict = {'你好': '你好', '谢谢': '不客气', '类别': '本模型的图像分割类别共有六种。'}
        response = ""
        word_list = word_split(prompt)
        for word in word_list:
            if word in my_dict:
                response += my_dict[word]
        if len(response) == 0:
            # with model.chat_session():
            #     response = model.generate(prompt, max_tokens=1024)
            client = OpenAI(
                # 控制台获取key和secret拼接，假使APIKey是key123456，APISecret是secret123456
                api_key="7d3e03a37aeb2cb3056f0bc557881e9a:ODQ2MTE2N2Q4N2U3NDkwNzA1N2ZjNGU1",
                base_url='https://spark-api-open.xf-yun.com/v1'  # 指向讯飞星火的请求地址
            )
            completion = client.chat.completions.create(
                model='general',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            # response = "您输入的信息未找到相关信息，请您重新输入"
            response = completion.choices[0].message.content
        with st.chat_message("assistant"):
            content = st.write_stream(generate_stream(response))
        st.session_state.messages.append({'role': 'assistant', 'content': content})

if __name__ == '__main__':
    st.header("图像分割")
    st.sidebar.header("请选择你的标签页")

    model = load_model("models/satellite_standard_unet_100epochs_7May2021.hdf5", compile=False)

    option = st.sidebar.selectbox("", options=['Image upload', 'Image Processing', 'Model predictions', 'Chatbots'])
    if option == 'Image upload':
        upload_image()
    elif option == 'Image Processing':
        process_image()
    elif option == 'Model predictions':
        model_predict(model=model)
    elif option == 'Chatbots':
        chatbot()
