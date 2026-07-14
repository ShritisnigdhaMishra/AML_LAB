from flask import Flask,render_template,request
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import io
import base64
app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="GET":
        return render_template("index.html",equation=None,chart_image=None)
    x_text=request.form["x_values"]
    y_text=request.form["y_values"]
    x=[float(i) for i in x_text.split(",")]
    y=[float(i) for i in y_text.split(",")]
    n=len(x)
    sum_x=sum(x)
    sum_y=sum(y)
    sum_xy=sum(x[i]*y[i] for i in range(n))
    sum_x2=sum(x[i]**2 for i in range(n))
    m=(n*sum_xy-sum_x*sum_y)/(n*sum_x2-sum_x**2)
    c=(sum_y-m*sum_x)/n
    y_pred=[m*xi+c for xi in x]
    errors=[y[i]-y_pred[i] for i in range(n)]
    mae=sum(abs(e) for e in errors)/n
    mse=sum(e**2 for e in errors)/n
    rmse=mse**0.5
    y_mean=sum_y/n
    ss_total=sum((yi-y_mean)**2 for yi in y)
    ss_residual=sum(e**2 for e in errors)
    r2=1-(ss_residual/ss_total)
    chart_image=make_charts(x,y,y_pred,mae,mse,rmse,r2)
    return render_template(
        "index.html",
        x_values=x_text,
        y_values=y_text,
        equation=f"y={m:.2f}x+{c:.2f}",
        chart_image=chart_image,
    )
def make_charts(x,y,y_pred,mae,mse,rmse,r2):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,9))
    ax1.scatter(x,y,color="green",label="Actual data",zorder=3)
    ax1.plot(x,y_pred,color="red",linewidth=2,label="Regression line")
    ax1.set_title("Regression Line")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    labels=["MAE","MSE","RMSE","R2"]
    values=[mae,mse,rmse,r2]
    ax2.scatter(labels,values,color="blue",s=120,zorder=3)
    for i,v in enumerate(values):
        ax2.annotate(f"{v:.2f}",(i, v),textcoords="offset points", xytext=(0, 10),
                ha="center",fontsize=10,fontweight="bold")
    ax2.set_title("Model Metrics Comparison")
    ax2.set_xlabel("Evaluation Metrics")
    ax2.set_ylabel("Values")
    ax2.grid(True,linestyle="--",alpha=0.5)
    plt.tight_layout()
    buffer=io.BytesIO()
    plt.savefig(buffer,format="png")
    plt.close(fig)
    buffer.seek(0)
    image_base64=base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64
if __name__=="__main__":
    app.run(debug=True)