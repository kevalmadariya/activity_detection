import uvicorn
import controller

def main():
    print("--- Starting X3D Video API ---")
    uvicorn.run(controller.app, host="0.0.0.0", port=8700)

if __name__ == "__main__":
    main()