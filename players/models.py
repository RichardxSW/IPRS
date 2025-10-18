from django.db import models

# Create your models here.
class Test(models.Model):
    tes=models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # unique_together = ('tes')
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.tes}"
    

class Dataset(models.Model):
    league_name=models.CharField(max_length=50)
    season=models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('league_name', 'season')
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.league_name} ({self.season})"
    

class Player(models.Model):
    dataset=models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='players')
    player_name=models.CharField(max_length=100)
    team=models.CharField(max_length=50, blank=True, null=True)
    position=models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering=['-uploaded_at']

    def __str__(self):
        return f"{self.player_name}"
