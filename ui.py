import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("🧠 Face Entry System Frontend")

choice = st.sidebar.radio("Choose Operation", [
    "Register User",
    "Delete User",
    "Start Face Verification",
    "Stop Face Verification"
])

# 1️⃣ REGISTER USER
if choice == "Register User":
    st.header("📸 Register User")
    name = st.text_input("Name")
    phone = st.text_input("Phone Number")
    image_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

    if st.button("Submit"):
        if not all([name, phone, image_file]):
            st.warning("All fields are required.")
        else:
            files = {"image": image_file}
            data = {"name": name, "phone": phone}
            res = requests.post(f"{API_URL}/create_user", data=data, files=files)

            if res.status_code == 201:
                st.success(res.json().get("message"))
            else:
                st.error(res.json().get("error", "Something went wrong."))

# 2️⃣ DELETE USER
elif choice == "Delete User":
    st.header("🗑️ Delete User")
    name = st.text_input("Name")
    phone = st.text_input("Phone Number")

    if st.button("Delete"):
        payload = {"name": name, "phone": phone}
        res = requests.post(f"{API_URL}/delete_user", json=payload)

        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error(res.json().get("detail", "User not found or already deleted."))

# 3️⃣ START VERIFICATION
elif choice == "Start Face Verification":
    st.header("🟢 Start Verification Thread")
    name = st.text_input("Thread Name", value="gate")
    is_visualize = st.checkbox("Visualize", value=True)

    if st.button("Start"):
        payload = {"name": name, "is_visualize": is_visualize}
        res = requests.post(f"{API_URL}/active/face_verify_agent", json=payload)

        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error(res.json().get("error", "Failed to start."))

# 4️⃣ STOP VERIFICATION
elif choice == "Stop Face Verification":
    st.header("🛑 Stop Verification Thread")
    name = st.text_input("Thread Name", value="gate")

    if st.button("Stop"):
        payload = {"name": name}
        res = requests.post(f"{API_URL}/stop/face_verify_agent", json=payload)

        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error(res.json().get("error", "No such thread running."))
