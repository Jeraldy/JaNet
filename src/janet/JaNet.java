/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package janet;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

/**
 *
 * @author Deus
 */
public class JaNet extends Application {
    
    NN nn = new NN();
    LineChart<Number, Number> lineChart;
    
    @Override
    public void start(Stage primaryStage) {
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis(0, 0.5, 0.1);
        
        xAxis.setLabel("Iterations");
        yAxis.setLabel("Loss");
        
        lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Training Loss");
        
        Button btn = new Button();
        btn.setText("Train");
        btn.setOnAction((e) -> {
            lineChart.getData().add(nn.train());
        });
        
        BorderPane root = new BorderPane();
        root.setCenter(lineChart);
        root.setRight(hyperParametersPane());
        root.setTop(btn);
        
        Scene scene = new Scene(root, 1000, 600);
        scene.getStylesheets().add(getClass().getResource("LineChart.css").toExternalForm());
        
        primaryStage.setTitle("Hello NN!");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
    private VBox hyperParametersPane() {
        VBox vbox = new VBox();
        vbox.setPrefWidth(300); 
        vbox.setPadding(new Insets(5,5,5,5));
        
        TextField txtLearningRate = new TextField();
        Label lbLearningRate = new Label("Learning Rate");
        
        TextField txtEpoch = new TextField();
        Label lbEpoch = new Label("Epochs");
        
        TextField txtMomentum = new TextField();
        Label lbMomentum = new Label("Momentum");
        
        TextField txtBatchSize = new TextField();
        Label lbBatchSize = new Label("Batch Size");
        
        vbox.getChildren().addAll(
                new HBox(10,lbLearningRate,txtLearningRate),
                new HBox(10,lbEpoch,txtEpoch),
                new HBox(10,lbMomentum,txtMomentum),
                new HBox(10,lbBatchSize,txtBatchSize)
        );
        
        return vbox;
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }
    
}
