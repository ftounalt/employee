import streamlit as st
import pickle
gb=pickle.load(open('gb_model.pkl','rb'))

def classify(num):
    if num==1:
        return 'The Employee will leave'
    else:
        return 'The Employee will Stay'
    
def main():
    st.title("Machine Learning Model Deployment")
    html_temp = """
    <div style="background-color:gray ;padding:10px">
    <h2 style="color:white;text-align:center;">Employee Future Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    sl = st.selectbox('Select you education(0- Bachelor, 1- Master, 2- PHD)', [i for i in range(3)])
    s2 = st.text_input('What is the joining year', value='0')
    s3 = st.selectbox('Select the city(0- Bangalore, 1- New delhi, 2- Pune)', [i for i in range(3)])
    s4 = st.selectbox('Select the PaymentTier(1- High payment tier,2- Medium payment tier,3- Low payment tier)', [ i+1 for i in range(3)])
    s5 = st.text_input('age', value='0')
    s6 = st.selectbox('gender(0- Female ,1- Male)', [i for i in range(2)])
    s7 = st.selectbox('EverBenched(0- No ,1- Yes)', [i for i in range(2)])
    s8 = st.text_input('Experience In Current Domain', value='0')
    inputs=[[sl,s2,s3,s4,s5,s6,s7,s8]]
    if st.button('Predict'):
        st.success(classify(gb.predict(inputs)))


if __name__=='__main__':
    main()